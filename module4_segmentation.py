"""
CrossSafe - Module 4: Scene Segmentation  (v4 - stable rewrite)
----------------------------------------------------------------
Key fixes in v4:
  - process() no longer has bare 'import torch' at top — was silently
    failing and locking segmentation to geometric fallback forever
  - Torch imports moved to load_segmentation_model() only
  - _cityscapes_to_zones() receives cs_up already at full resolution
    (interpolation done inside process() with torch, or cv2 fallback)
  - Morphological open kernel reduced 15→9 (was erasing all road)
  - Bonnet mask: bottom 10% (was 12%, too aggressive for some videos)
  - DNN_EVERY_N reset to 8 — stable cache, run every 8 frames
  - ALPHA_OVERLAY = 0.12 — subtle green tint only

Cityscapes label mapping:
  road(0) → ZONE_ROAD
  sidewalk(1), terrain(9) → ZONE_FOOTPATH
  road markings via zebra detector → ZONE_CROSSWALK
"""

import cv2
import numpy as np
import time

# ─── CONFIG ────────────────────────────────────────────────
SEG_WIDTH      = 320
SEG_HEIGHT     = 320
ALPHA_OVERLAY  = 0.12
DANGER_ALPHA   = 0.28
DNN_EVERY_N    = 8

# Cityscapes class indices
CS_ROAD       = 0
CS_SIDEWALK   = 1
CS_TERRAIN    = 9

# Zone IDs
ZONE_ROAD      = 0
ZONE_FOOTPATH  = 1
ZONE_CROSSWALK = 2
ZONE_OTHER     = 3

ZONE_COLORS = {
    ZONE_ROAD      : (0,  160,  0),
    ZONE_FOOTPATH  : (160,  60,  0),
    ZONE_CROSSWALK : (0,  200, 200),
    ZONE_OTHER     : None,
}
DANGER_COLOR = (0, 0, 200)


# ═══════════════════════════════════════════════════════════
#  MODEL LOADER  — all torch imports contained here
# ═══════════════════════════════════════════════════════════

def load_segmentation_model():
    """
    Load SegFormer-b0-cityscapes.
    ALL torch/transformers imports are inside this function.
    Returns (processor, model, device, torch_module).
    The torch_module is returned so callers don't need to import torch.
    """
    from transformers import SegformerForSemanticSegmentation
    import torch
    import torch.nn.functional as F

    # SegformerImageProcessor requires transformers>=4.19
    # AutoImageProcessor works in all versions >=4.11 — use as fallback
    try:
        from transformers import SegformerImageProcessor
        _Processor = SegformerImageProcessor
    except ImportError:
        from transformers import AutoImageProcessor
        _Processor = AutoImageProcessor

    print("Loading SegFormer-b0-cityscapes (~14MB, cached after first run)...")
    model_name = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
    processor  = _Processor.from_pretrained(model_name)
    model      = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    print(f"  SegFormer loaded on: {device}")
    return processor, model, device, torch


# ═══════════════════════════════════════════════════════════
#  FISHEYE MASK
# ═══════════════════════════════════════════════════════════

def _make_fisheye_mask(h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2),
                (int(w * 0.48), int(h * 0.48)),
                0, 0, 360, 1, -1)
    return mask


# ═══════════════════════════════════════════════════════════
#  ZONE EXTRACTOR
# ═══════════════════════════════════════════════════════════

def _build_zone_map(cs_up_full, h, w, fisheye_mask=None):
    """
    cs_up_full: (h, w) int array with Cityscapes class indices at full resolution.
    Returns zone_map (h, w) uint8.
    """
    zone_map = np.full((h, w), ZONE_OTHER, dtype=np.uint8)

    zone_map[cs_up_full == CS_ROAD]                          = ZONE_ROAD
    zone_map[(cs_up_full == CS_SIDEWALK) |
             (cs_up_full == CS_TERRAIN)]                     = ZONE_FOOTPATH

    # ── Post-processing ────────────────────────────────────

    # 1. Mask bottom 10% — GoPro bonnet/hood reflection
    zone_map[int(h * 0.90):, :] = ZONE_OTHER

    # 2. Morphological open — remove stray thin road bleed at frame edges
    #    Kernel 9×9: aggressive enough to kill bollard-area bleed,
    #    conservative enough not to erase the main road body
    road_mask  = (zone_map == ZONE_ROAD).astype(np.uint8)
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    road_clean = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
    zone_map[zone_map == ZONE_ROAD] = ZONE_OTHER
    zone_map[road_clean == 1]       = ZONE_ROAD

    # 3. Fisheye corner mask
    if fisheye_mask is not None:
        zone_map[fisheye_mask == 0] = ZONE_OTHER

    return zone_map


# ═══════════════════════════════════════════════════════════
#  GEOMETRIC FALLBACK
# ═══════════════════════════════════════════════════════════

def _geometric_zones(h, w, fisheye_mask=None):
    """Perspective-correct trapezoid road — used when DNN unavailable."""
    zone_map   = np.full((h, w), ZONE_OTHER, dtype=np.uint8)
    cx         = w // 2
    road_top_y = int(h * 0.62)
    road_top_w = int(w * 0.28)
    pts = np.array([
        [cx - road_top_w, road_top_y],
        [cx + road_top_w, road_top_y],
        [w,               int(h * 0.90)],
        [0,               int(h * 0.90)],
    ], dtype=np.int32)
    cv2.fillPoly(zone_map, [pts], ZONE_ROAD)
    if fisheye_mask is not None:
        zone_map[fisheye_mask == 0] = ZONE_OTHER
    return zone_map


# ═══════════════════════════════════════════════════════════
#  ZEBRA CROSSING DETECTOR
# ═══════════════════════════════════════════════════════════

def detect_zebra_crossing(frame_gray, zone_map):
    road_m = (zone_map == ZONE_ROAD)
    if not road_m.any():
        return zone_map
    sobel    = np.abs(cv2.Sobel(frame_gray, cv2.CV_32F, 0, 1, ksize=3))
    _, thr   = cv2.threshold(sobel, 100, 1, cv2.THRESH_BINARY)
    thr      = thr.astype(np.uint8) * road_m.astype(np.uint8)
    row_den  = thr.mean(axis=1)
    rows     = np.where(row_den > 0.10)[0]
    if len(rows) < 8:
        return zone_map
    for block in np.split(rows, np.where(np.diff(rows) > 15)[0] + 1):
        if len(block) >= 8 and (block[-1] - block[0]) >= 20:
            zone_map[block[0]:block[-1]+1, :][
                zone_map[block[0]:block[-1]+1, :] == ZONE_ROAD
            ] = ZONE_CROSSWALK
    return zone_map


# ═══════════════════════════════════════════════════════════
#  SCENE SEGMENTER
# ═══════════════════════════════════════════════════════════

class SceneSegmenter:

    def __init__(self, use_dnn=True, fisheye=True):
        self.use_dnn      = use_dnn
        self.fisheye      = fisheye
        self.processor    = None
        self.model        = None
        self.device       = 'cpu'
        self._torch       = None   # torch module held here — no global import
        self.zone_map     = None
        self._last_seg    = None
        self._frame_count = 0
        self._fish_mask   = None

        if use_dnn:
            try:
                self.processor, self.model, self.device, self._torch = \
                    load_segmentation_model()
            except Exception as e:
                print(f"[Module4] DNN load failed: {e}")
                print("[Module4] Falling back to geometric mode.")
                self.use_dnn = False

    def _get_fisheye_mask(self, h, w):
        if self._fish_mask is None or self._fish_mask.shape != (h, w):
            self._fish_mask = _make_fisheye_mask(h, w) if self.fisheye else None
        return self._fish_mask

    def process(self, frame):
        """
        Run segmentation every DNN_EVERY_N frames, cache in between.
        NO torch imports here — all done in load_segmentation_model().
        """
        h, w = frame.shape[:2]
        self._frame_count += 1
        fish_mask = self._get_fisheye_mask(h, w)

        dnn_ready = self.use_dnn and self.model is not None and self._torch is not None
        run_dnn   = dnn_ready and (
            self._last_seg is None or
            self._frame_count % DNN_EVERY_N == 1   # frame 1, 9, 17... (offset avoids 0)
        )

        if run_dnn:
            try:
                torch = self._torch
                small = cv2.resize(frame, (SEG_WIDTH, SEG_HEIGHT))
                rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                inputs = self.processor(images=rgb, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)

                # logits shape: (1, 19, H/4, W/4)
                logits = outputs.logits
                # Upsample to full frame size
                import torch.nn.functional as F
                cs_up = F.interpolate(
                    logits, size=(h, w), mode='bilinear', align_corners=False
                ).argmax(dim=1).squeeze().cpu().numpy().astype(np.int32)

                self._last_seg = _build_zone_map(cs_up, h, w, fish_mask)
                print(f"  [M4] DNN zone map updated at frame {self._frame_count} "
                      f"road_px={int((self._last_seg==ZONE_ROAD).sum())}")

            except Exception as e:
                print(f"  [M4] DNN inference error: {e} — using geometric")
                self._last_seg = _geometric_zones(h, w, fish_mask)

        elif self._last_seg is None:
            # First frame, DNN not ready
            self._last_seg = _geometric_zones(h, w, fish_mask)

        zone_map = self._last_seg.copy()
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        zone_map = detect_zebra_crossing(gray, zone_map)
        self.zone_map = zone_map
        return zone_map

    def draw_overlay(self, display, zone_map, person_boxes=None):
        h, w = display.shape[:2]

        overlay = display.copy()
        for zone_id, color in ZONE_COLORS.items():
            if color is None:
                continue
            mask = (zone_map == zone_id)
            if not mask.any():
                continue
            overlay[mask] = color
        cv2.addWeighted(overlay, ALPHA_OVERLAY, display, 1 - ALPHA_OVERLAY, 0, display)

        if person_boxes:
            dang = display.copy()
            for box in person_boxes:
                x1, y1, x2, y2 = box[:4]
                feet_y = min(y1 + int((y2 - y1) * 0.75), h - 1)
                cx_p   = max(0, min((x1 + x2) // 2, w - 1))
                if zone_map[feet_y, cx_p] == ZONE_ROAD:
                    cv2.rectangle(dang, (x1, y1), (x2, y2), DANGER_COLOR, -1)
            cv2.addWeighted(dang, DANGER_ALPHA, display, 1 - DANGER_ALPHA, 0, display)

        # Legend
        legend = [
            (ZONE_COLORS[ZONE_ROAD],      'Road'),
            (ZONE_COLORS[ZONE_FOOTPATH],  'Footpath'),
            (ZONE_COLORS[ZONE_CROSSWALK], 'Crosswalk'),
            (DANGER_COLOR,                'Ped danger'),
        ]
        h, w = display.shape[:2]
        for i, (lc, lt) in enumerate(legend):
            oy = h - 90 + i * 20
            cv2.rectangle(display, (6, oy), (18, oy + 12), lc, -1)
            cv2.putText(display, lt, (22, oy + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1)

    def is_in_road(self, x1, y1, x2, y2):
        if self.zone_map is None:
            return False
        h, w   = self.zone_map.shape
        feet_y = min(y1 + int((y2 - y1) * 0.75), h - 1)
        cx     = max(0, min((x1 + x2) // 2, w - 1))
        return int(self.zone_map[feet_y, cx]) == ZONE_ROAD

    def is_near_crosswalk(self, x1, y1, x2, y2, margin=60):
        if self.zone_map is None:
            return False
        h, w   = self.zone_map.shape
        region = self.zone_map[
            max(0, y1 - margin):min(h, y2 + margin),
            max(0, x1 - margin):min(w, x2 + margin)
        ]
        return bool((region == ZONE_CROSSWALK).any())


# ═══════════════════════════════════════════════════════════
#  STANDALONE DEMO
# ═══════════════════════════════════════════════════════════

def run_demo(source=0):
    seg  = SceneSegmenter(use_dnn=True, fisheye=True)
    cap  = cv2.VideoCapture(source)
    prev = time.time()
    print("Module 4 demo — Q=quit | O=overlay | F=fisheye")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display  = frame.copy()
        zone_map = seg.process(frame)
        seg.draw_overlay(display, zone_map)
        fps = 1 / (time.time() - prev + 0.001)
        prev = time.time()
        h, w = display.shape[:2]
        cv2.rectangle(display, (0, 0), (w, 52), (15, 15, 15), -1)
        cv2.putText(display, f'CrossSafe M4 v4  FPS:{fps:.1f}  '
                    f'Mode:{"DNN" if seg.use_dnn else "Geo"}',
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2)
        cv2.imshow('CrossSafe - Module 4', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import sys
    run_demo(sys.argv[1] if len(sys.argv) > 1 else 0)