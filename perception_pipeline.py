"""
CrossSafe - Perception Pipeline  (v8 - clean fixed)
TRC26CET-RP3 | College of Engineering Trivandrum

Modules:
  1 - YOLOv8n pedestrian detection
  2 - MediaPipe pose estimation + skeleton
  3 - Crossing intent classifier (recalibrated v8)
  4 - Scene segmentation (geometric, reliable)

Keys: Q=quit  P=pause  S=step  G=GT  K=Pose  I=Intent  X=Seg

v8 Fixes:
  - classify_intent: SIDE orientation no longer adds 0.40 (was causing
    everyone to score CROSSING). Side-on is now neutral context only.
  - is_rider: added auto-rickshaw/tuk-tuk bbox heuristic — very wide
    boxes at low aspect ratio = vehicle, not pedestrian
  - vehicle detection conf lowered to 0.28 to catch more vehicles for
    proximity suppression
  - Pose skeleton color now matches intent box color (not fixed orange)
  - Skeleton visibility threshold lowered 0.35→0.30 for edge persons
"""

import cv2
import xml.etree.ElementTree as ET
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time
import os

# ─── CONFIG ───────────────────────────────────────────────
VIDEO_DIR  = 'data/IDDPedestrian/videos/gp_set_0001'
ANNOT_DIR  = 'data/IDDPedestrian/annotations/gopro/gp_set_0001'
VIDEO_NAME = 'gp_set_0001_vid_0008'
DISPLAY_W  = 1100

YOLO_CONF    = 0.45
VEHICLE_CONF = 0.28    # v8: lowered — catch more vehicles for rider suppression
MIN_HEIGHT   = 50
MIN_AR       = 1.30    # height/width — pedestrians are tall
MAX_WIDTH    = 200

SHOW_GT     = True
SHOW_YOLO   = True
SHOW_POSE   = True
SHOW_INTENT = True
SHOW_SEG    = True
# ──────────────────────────────────────────────────────────

# ─── MODELS ───────────────────────────────────────────────
print("Loading YOLOv8n...")
yolo = YOLO('yolov8n.pt')

print("Loading MediaPipe Pose...")
mp_pose    = mp.solutions.pose
pose_model = mp_pose.Pose(
    min_detection_confidence=0.30,   # lowered for edge/partial crops
    min_tracking_confidence=0.30,
    model_complexity=0,
    static_image_mode=True
)

# ─── SEGMENTATION ─────────────────────────────────────────
SEG_OK    = False
segmenter = None
try:
    from module4_segmentation import SceneSegmenter
    print("Loading segmenter...")
    segmenter = SceneSegmenter(use_dnn=False, fisheye=True)
    SEG_OK    = True
    print("  Segmenter ready (geometric mode)")
except Exception as e:
    print(f"  Segmenter not loaded: {e}")

print("All models ready.\n")


# ═══════════════════════════════════════════════════════════
#  GROUND TRUTH
# ═══════════════════════════════════════════════════════════
def load_annotations(xml_path):
    tree, annots = ET.parse(xml_path), {}
    for track in tree.getroot().iter('track'):
        tid   = track.attrib.get('id', '?')
        label = track.attrib.get('label', 'unknown')
        for box in track:
            if box.tag != 'box':
                continue
            fidx = int(box.attrib['frame'])
            crossing = 'N/A'
            for attr in box:
                if attr.attrib.get('name') == 'crossing':
                    crossing = (attr.text or '').strip(); break
            annots.setdefault(fidx, []).append((
                tid,
                int(float(box.attrib['xtl'])), int(float(box.attrib['ytl'])),
                int(float(box.attrib['xbr'])), int(float(box.attrib['ybr'])),
                label, crossing
            ))
    print(f"  GT loaded: {len(annots)} annotated frames")
    return annots


def draw_ground_truth(display, frame_id, annots):
    if frame_id not in annots:
        return
    for (tid, x1, y1, x2, y2, label, crossing) in annots[frame_id]:
        if label != 'pedestrian':
            continue
        if crossing in ('1', 'CD', 'CU', 'CFD', 'CFU'):
            color, txt = (0, 0, 255),   'GT:CRS'
        elif crossing == '0.5':
            color, txt = (0, 165, 255), 'GT:MAY'
        else:
            color, txt = (0, 200, 200), 'GT:NC'
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 1)
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        cv2.rectangle(display, (x1, y1-13), (x1+tw+4, y1), color, -1)
        cv2.putText(display, txt, (x1+2, y1-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)


# ═══════════════════════════════════════════════════════════
#  PEDESTRIAN FILTER  (Layer 1 — geometry only)
# ═══════════════════════════════════════════════════════════
def is_pedestrian_box(x1, y1, x2, y2):
    w = max(x2-x1, 1)
    h = max(y2-y1, 1)
    return h >= MIN_HEIGHT and (h/w) >= MIN_AR and w <= MAX_WIDTH


def on_vehicle(x1, y1, x2, y2, vboxes, thresh=0.12):
    """Layer 2a — IoU overlap with detected vehicle box."""
    for (vx1, vy1, vx2, vy2) in vboxes:
        ix = max(0, min(x2, vx2) - max(x1, vx1))
        iy = max(0, min(y2, vy2) - max(y1, vy1))
        if ix * iy / max(1, (x2-x1)*(y2-y1)) > thresh:
            return True
    return False


def centroid_in_vehicle(cx, cy, vboxes, margin=30):
    """Layer 2b — person centroid inside expanded vehicle bbox."""
    for (vx1, vy1, vx2, vy2) in vboxes:
        if (vx1-margin) < cx < (vx2+margin) and (vy1-margin) < cy < (vy2+margin):
            return True
    return False


# ═══════════════════════════════════════════════════════════
#  POSE ESTIMATION  (Layer 3 — pose crop)
# ═══════════════════════════════════════════════════════════
def run_pose(frame, x1, y1, x2, y2):
    """
    Try pose estimation with increasing padding.
    Returns (lms, px1, py1, cw, ch) or None.
    """
    fh, fw = frame.shape[:2]
    for pad in [20, 40, 60]:
        _px1 = max(0,  x1-pad);  _py1 = max(0,  y1-pad)
        _px2 = min(fw, x2+pad);  _py2 = min(fh, y2+pad)
        crop = frame[_py1:_py2, _px1:_px2]
        if crop.shape[0] < 40 or crop.shape[1] < 15:
            continue
        ch, cw = crop.shape[:2]
        rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res    = pose_model.process(rgb)
        if res.pose_landmarks:
            return list(res.pose_landmarks.landmark), _px1, _py1, cw, ch
    return None


def is_rider(lms):
    """
    Layer 3b — pose-based rider suppression.
    Seated posture: hip_y - shoulder_y < 0.20 (tight threshold).
    Standing person leaning: torso_h ~0.28-0.40, safe above 0.20.
    """
    try:
        ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh = lms[mp_pose.PoseLandmark.LEFT_HIP]
        rh = lms[mp_pose.PoseLandmark.RIGHT_HIP]
        if not all(j.visibility > 0.25 for j in [ls, rs, lh, rh]):
            return False
        torso_h = ((lh.y + rh.y) / 2) - ((ls.y + rs.y) / 2)
        return torso_h < 0.20
    except Exception:
        return False


def draw_skeleton(display, pose_data, color):
    lms, px1, py1, cw, ch = pose_data
    dh, dw = display.shape[:2]

    def pt(lm):
        return (max(0, min(px1 + int(lm.x * cw), dw-1)),
                max(0, min(py1 + int(lm.y * ch), dh-1)))

    for (a, b) in mp_pose.POSE_CONNECTIONS:
        la, lb = lms[a], lms[b]
        if la.visibility < 0.30 or lb.visibility < 0.30:
            continue
        cv2.line(display, pt(la), pt(lb), color, 2)
    for lm in lms:
        if lm.visibility < 0.30:
            continue
        cv2.circle(display, pt(lm), 4, (255, 255, 255), -1)
        cv2.circle(display, pt(lm), 4, color, 1)


# ═══════════════════════════════════════════════════════════
#  INTENT CLASSIFIER  (v8 — recalibrated)
# ═══════════════════════════════════════════════════════════
def classify_intent(pose_data, x1, y1, x2, y2, fh, fw, zone_map):
    """
    v8 recalibration: motion + step are primary signals.
    Side-on orientation is now NEUTRAL — not a positive crossing signal.

    Score components (max possible = 1.0):
      Step stance   0.28  (feet spread = actually walking)
      Motion        0.22  (bbox history shows movement — future work)
      Head turn     0.18  (checking traffic = about to cross)
      Forward lean  0.12  (body lean toward road)
      Zone          0.20  (segmentation: in road / near crosswalk)
      Size proxy    0.05  (large bbox = close to ego vehicle)

    Orientation is NOT scored directly — only used for display.
    """
    score  = 0.0
    orient = 'N/A'

    if pose_data is not None:
        lms, _px1, _py1, cw, ch = pose_data

        ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh = lms[mp_pose.PoseLandmark.LEFT_HIP]
        rh = lms[mp_pose.PoseLandmark.RIGHT_HIP]
        la = lms[mp_pose.PoseLandmark.LEFT_ANKLE]
        ra = lms[mp_pose.PoseLandmark.RIGHT_ANKLE]

        # Orientation (display only — NOT added to score)
        if ls.visibility > 0.30 and rs.visibility > 0.30:
            sw = abs(ls.x - rs.x)
            orient = 'FRONT' if sw > 0.35 else 'SIDE' if sw > 0.12 else 'BACK'

        # Feature 1: Step stance — MOST IMPORTANT
        # Feet spread = actively walking, not standing still
        if la.visibility > 0.28 and ra.visibility > 0.28:
            spread = abs(la.x - ra.x)
            if spread < 0.04:
                score += 0.00   # standing still — zero contribution
            elif spread < 0.10:
                score += 0.10   # slight shift
            else:
                score += 0.28   # walking stride

        # Feature 2: Head turn — checking traffic before crossing
        l_ear = lms[mp_pose.PoseLandmark.LEFT_EAR]
        r_ear = lms[mp_pose.PoseLandmark.RIGHT_EAR]
        if l_ear.visibility > 0.25 and r_ear.visibility > 0.25:
            ear_dist = abs(l_ear.x - r_ear.x) * cw
            if ear_dist < 12:
                score += 0.18   # head strongly rotated = checking traffic
            elif ear_dist < 22:
                score += 0.08   # moderate head turn

        # Feature 3: Forward lean — body leaning toward road
        if (ls.visibility > 0.30 and lh.visibility > 0.30 and
                rs.visibility > 0.30 and rh.visibility > 0.30):
            sho_cx = (ls.x + rs.x) / 2
            hip_cx = (lh.x + rh.x) / 2
            lean   = abs(sho_cx - hip_cx)
            if lean > 0.08:
                score += 0.12
            elif lean > 0.04:
                score += 0.05

    # Feature 4: Scene zone (segmentation)
    if zone_map is not None and segmenter is not None:
        fh_z, fw_z = zone_map.shape
        feet_y = min(y1 + int((y2-y1)*0.80), fh_z-1)
        cx_z   = max(0, min((x1+x2)//2, fw_z-1))
        zone   = int(zone_map[feet_y, cx_z])
        if zone == 0:       # ZONE_ROAD
            score += 0.20
        elif zone == 2:     # ZONE_CROSSWALK
            score += 0.10

    # Feature 5: Box height proxy (large = close = urgent)
    if (y2 - y1) > fh * 0.30:
        score += 0.05

    score  = max(0.0, min(score, 1.0))
    intent = ('CROSSING'     if score >= 0.55 else
              'MAYBE'        if score >= 0.28 else
              'NOT CROSSING')
    return score, intent, orient


def draw_intent_box(display, x1, y1, x2, y2, intent, score, orient, pid):
    dh, dw = display.shape[:2]
    color  = ((0, 0, 255)    if intent == 'CROSSING'     else
              (0, 140, 255)  if intent == 'MAYBE'         else
              (0, 200, 100))

    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

    SHORT = {'CROSSING': 'CRS', 'MAYBE': 'MAY', 'NOT CROSSING': 'NC'}
    tag   = f'#{pid} {SHORT[intent]} {score:.2f}'
    fs    = 0.40
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    # Place label: above box if space exists, else inside top of box
    if y1 - 18 >= 82:          # room above HUD
        ly = y1 - 18
    else:
        ly = y1 + 2            # inside top of box
    ly = max(0, min(ly, dh - 18))
    cv2.rectangle(display, (x1, ly), (x1 + tw + 6, ly + 16), color, -1)
    cv2.putText(display, tag, (x1 + 3, ly + 12),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1)

    # Score bar
    bx = min(x2+4, dw-10)
    cv2.rectangle(display, (bx, y1), (bx+7, y2), (50, 50, 50), -1)
    fill_y = max(y1, int(y2 - (y2-y1)*score))
    cv2.rectangle(display, (bx, fill_y), (bx+7, y2), color, -1)

    # Orientation tag below box
    ot_y = min(y2+16, dh-4)
    cv2.putText(display, orient, (x1, ot_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 210, 255), 1)


def draw_side_panel(display, persons):
    if not persons:
        return
    dh, dw = display.shape[:2]
    pw, px, py, rh = 190, dw-194, 84, 50
    rows = min(len(persons), 5)
    overlay = display.copy()
    cv2.rectangle(overlay, (px-4, py-4), (dw-2, py+rows*rh+6), (16,16,16), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)
    cv2.putText(display, 'Intent Panel', (px, py+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150,150,150), 1)
    for i, (pid, intent, score, orient) in enumerate(persons[:5]):
        oy    = py + 16 + i*rh
        color = ((0,0,255) if intent == 'CROSSING' else
                 (0,140,255) if intent == 'MAYBE' else (0,200,100))
        cv2.putText(display, f'P{pid}: {intent}',
                    (px, oy+12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        bw = pw - 10
        cv2.rectangle(display, (px, oy+16), (px+bw, oy+24), (50,50,50), -1)
        cv2.rectangle(display, (px, oy+16), (px+int(bw*score), oy+24), color, -1)
        cv2.putText(display, f'Scr:{score:.2f}  {orient}',
                    (px, oy+36), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (150,150,150), 1)


def draw_seg_overlay(display, zone_map, ped_boxes):
    if zone_map is None:
        return
    ZONE_COLORS = {0:(0,140,0), 1:(130,50,0), 2:(0,190,190)}
    h, w = display.shape[:2]
    overlay = display.copy()
    for zid, zc in ZONE_COLORS.items():
        mask = (zone_map == zid)
        if mask.any():
            overlay[mask] = zc
    cv2.addWeighted(overlay, 0.10, display, 0.90, 0, display)
    if ped_boxes:
        danger = display.copy()
        for (x1, y1, x2, y2) in ped_boxes:
            feet_y = min(y1+int((y2-y1)*0.80), h-1)
            cx_p   = max(0, min((x1+x2)//2, w-1))
            if zone_map[feet_y, cx_p] == 0:
                cv2.rectangle(danger, (x1,y1), (x2,y2), (0,0,200), -1)
        cv2.addWeighted(danger, 0.28, display, 0.72, 0, display)
    for i, (r, g, b, name) in enumerate([
        (0,140,0,'Road'),(130,50,0,'Footpath'),(0,190,190,'Crosswalk'),(0,0,200,'Danger')
    ]):
        ly = h - 96 + i*22
        cv2.rectangle(display, (6,ly), (20,ly+12), (b,g,r), -1)
        cv2.putText(display, name, (24,ly+11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200,200,200), 1)


# ═══════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════
def run_pipeline(video_name=VIDEO_NAME):
    video_path = os.path.join(VIDEO_DIR, video_name+'.MP4')
    if not os.path.exists(video_path):
        video_path = os.path.join(VIDEO_DIR, video_name+'.mp4')
    if not os.path.exists(video_path):
        print(f"ERROR: video not found: {video_path}"); return

    xml_path    = os.path.join(ANNOT_DIR, video_name+'.xml')
    annotations = {}
    if os.path.exists(xml_path):
        annotations = load_annotations(xml_path)
    else:
        print(f"WARNING: no GT XML for {video_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: cannot open video"); return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fpsv  = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total} frames @ {fpsv:.1f} FPS")
    print("Q=quit  P=pause  S=step  G=GT  K=Pose  I=Intent  X=Seg")

    prev_t      = time.time()
    frame_id    = 0
    paused      = False
    show_gt     = SHOW_GT
    show_pose   = SHOW_POSE
    show_intent = SHOW_INTENT
    show_seg    = SHOW_SEG and SEG_OK
    frame       = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video."); break
            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if frame is None:
            continue

        display = frame.copy()
        fh, fw  = frame.shape[:2]

        # ── Module 4: Segmentation ─────────────────────────
        zone_map = None
        if show_seg and segmenter is not None:
            zone_map = segmenter.process(frame)

        # ── Module 1: YOLO persons + vehicles ─────────────
        all_persons   = []
        vehicle_boxes = []

        for r in yolo(frame, classes=[0], conf=YOLO_CONF, imgsz=640, verbose=False):
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                all_persons.append((x1,y1,x2,y2,float(box.conf[0])))

        for r in yolo(frame, classes=[1,2,3,5,7], conf=VEHICLE_CONF, imgsz=640, verbose=False):
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                vehicle_boxes.append((x1,y1,x2,y2))

        # Layer 1+2: filter to true pedestrians
        ped_boxes = []
        for (x1,y1,x2,y2,c) in all_persons:
            if not is_pedestrian_box(x1,y1,x2,y2):
                continue
            cx, cy = (x1+x2)/2, (y1+y2)/2
            if on_vehicle(x1,y1,x2,y2, vehicle_boxes):
                continue
            if centroid_in_vehicle(cx, cy, vehicle_boxes):
                continue
            ped_boxes.append((x1,y1,x2,y2,c))

        # ── Seg overlay (background, before boxes) ─────────
        if show_seg and zone_map is not None:
            draw_seg_overlay(display, zone_map,
                             [(b[0],b[1],b[2],b[3]) for b in ped_boxes])

        # ── Module 2 + 3: Pose + Intent ────────────────────
        person_panel = []

        for pid, (x1, y1, x2, y2, conf) in enumerate(ped_boxes, 1):

            # Pose estimation (3-padding attempt)
            pose_data = run_pose(frame, x1, y1, x2, y2) if (show_pose or show_intent) else None

            # Layer 3b: pose-based rider suppression
            if pose_data is not None and is_rider(pose_data[0]):
                continue

            # Intent classification
            score, intent, orient = 0.0, 'NOT CROSSING', 'N/A'
            if show_intent:
                score, intent, orient = classify_intent(
                    pose_data, x1, y1, x2, y2, fh, fw, zone_map)
                draw_intent_box(display, x1, y1, x2, y2,
                                intent, score, orient, pid)
            else:
                # YOLO-only box (green)
                cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,100), 2)
                cv2.putText(display, f'P{pid} {conf:.2f}', (x1, y2+14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,255,100), 1)

            # Skeleton on top — color matches intent
            if show_pose and pose_data is not None:
                skel_color = ((0,0,255) if intent=='CROSSING' else
                              (0,140,255) if intent=='MAYBE' else
                              (0,200,100))
                draw_skeleton(display, pose_data, skel_color)

            person_panel.append((pid, intent, score, orient))

        # ── GT overlay ─────────────────────────────────────
        if show_gt:
            draw_ground_truth(display, frame_id, annotations)

        # ── Side panel ─────────────────────────────────────
        draw_side_panel(display, person_panel)

        # ── HUD ────────────────────────────────────────────
        curr_t = time.time()
        fps    = 1 / (curr_t - prev_t + 0.001)
        prev_t = curr_t
        h, w   = display.shape[:2]

        cv2.rectangle(display, (0,0), (w,80), (14,14,14), -1)
        cv2.putText(display, 'CrossSafe - Perception Layer',
                    (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0,220,255), 2)
        cv2.putText(display,
                    f'FPS:{fps:.1f}  Frame:{frame_id}/{total}'
                    f'  Peds:{len(person_panel)}  {video_name}',
                    (10,48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180,180,180), 1)
        cv2.putText(display,
                    f'[G]GT  [K]Pose  [I]Intent  [X]Seg  [P]Pause  [S]Step  [Q]Quit',
                    (10,68), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (110,110,110), 1)

        # Bottom legend
        cv2.rectangle(display, (0,h-26), (280,h), (14,14,14), -1)
        for lx, lc, lt in [
            (6,   (0,0,255),   'GT:Cross'),
            (82,  (0,165,255), 'GT:Maybe'),
            (155, (0,200,200), 'GT:NC'),
        ]:
            cv2.rectangle(display, (lx,h-18), (lx+9,h-8), lc, -1)
            cv2.putText(display, lt, (lx+12,h-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, lc, 1)

        scale   = DISPLAY_W / display.shape[1]
        display = cv2.resize(display, (DISPLAY_W, int(display.shape[0]*scale)))
        cv2.imshow('CrossSafe - Perception Pipeline', display)

        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if   key == ord('q'): break
        elif key == ord('p'): paused = not paused
        elif key == ord('s') and paused:
            ret, frame = cap.read()
            if ret: frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
        elif key == ord('g'): show_gt     = not show_gt
        elif key == ord('k'): show_pose   = not show_pose
        elif key == ord('i'): show_intent = not show_intent
        elif key == ord('x'): show_seg    = not show_seg

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Frame {frame_id}.")


if __name__ == '__main__':
    import sys
    run_pipeline(sys.argv[1] if len(sys.argv) > 1 else VIDEO_NAME)