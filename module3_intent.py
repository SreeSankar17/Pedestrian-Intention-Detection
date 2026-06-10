"""
CrossSafe - Module 3: Crossing Intent Classifier  (v7 - recalibrated)
----------------------------------------------------------------------
v7 Core fix: Complete scoring recalibration.

Root cause of "everyone = CROSSING" bug:
  - feat_orientation returned 0.85 for ANY side-on person (weight 0.28)
  - Dashcam footage always shows people side-on → everyone got high score
  - SIDE orientation alone was pushing score to 0.24 before any other feature
  - Combined with default motion/proximity defaults → most people > 0.60

v7 Fixes:
  1. feat_orientation: side-on is now NEUTRAL (0.50), not high (0.85)
     Only WALKING (ankle spread + motion) makes side-on meaningful
  2. feat_step: feet-together (standing still) now strongly penalises (0.05)
  3. Weights rebalanced: motion+step are now the primary signals (0.25+0.20)
     orientation demoted to secondary context (0.15)
  4. CROSSING threshold raised 0.60 → 0.65, MAYBE 0.35 → 0.42
  5. Rider suppression: compressed_torso threshold tightened 0.26 → 0.20
     (was too aggressive — standing people leaning forward were suppressed)
  6. Segmentation: transformers AutoImageProcessor fallback for old versions

Intent labels:
  CROSSING  score >= 0.65  → red
  MAYBE     0.42–0.65      → orange  
  NOT CROSS score <  0.42  → cyan
"""

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from collections import deque
import time

# ─── CONFIG ───────────────────────────────────────────────
CONF_THRESHOLD  = 0.48
HISTORY_LEN     = 16
CROSSING_THRESH = 0.65   # v7: raised from 0.60
MAYBE_THRESH    = 0.42   # v7: raised from 0.35
MIN_ASPECT      = 1.35
MAX_ASPECT      = 6.0
MIN_HEIGHT_PX   = 45
MAX_WIDTH_PX    = 210
VEH_PROXIMITY   = 50
# ──────────────────────────────────────────────────────────

mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(
    min_detection_confidence=0.30,
    min_tracking_confidence=0.30,
    model_complexity=0
)


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════

def _vis(lm, thr=0.40):
    return lm.visibility >= thr


def _is_pedestrian_box(x1, y1, x2, y2):
    w  = max(x2 - x1, 1)
    h  = max(y2 - y1, 1)
    ar = h / w
    if w > MAX_WIDTH_PX:
        return False
    return MIN_ASPECT <= ar <= MAX_ASPECT and h >= MIN_HEIGHT_PX


def _near_vehicle(cx, cy, vehicle_boxes):
    """Suppress if person centroid is inside an expanded vehicle box."""
    exp = VEH_PROXIMITY // 2
    for (vx1, vy1, vx2, vy2) in vehicle_boxes:
        if (vx1 - exp) < cx < (vx2 + exp) and (vy1 - exp) < cy < (vy2 + exp):
            return True
    return False


def _on_vehicle(px1, py1, px2, py2, vehicle_boxes):
    for (vx1, vy1, vx2, vy2) in vehicle_boxes:
        ix1 = max(px1, vx1); iy1 = max(py1, vy1)
        ix2 = min(px2, vx2); iy2 = min(py2, vy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter / max(1, (px2 - px1) * (py2 - py1)) > 0.12:
            return True
    return False


def _is_rider_pose(lms, box_w=None):
    """
    Rider suppression via pose.
    v7: compressed_torso threshold tightened 0.26 → 0.20
    Standing people leaning forward have torso_h ~0.28-0.35
    Seated scooter riders have torso_h ~0.10-0.18
    """
    try:
        ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh = lms[mp_pose.PoseLandmark.LEFT_HIP]
        rh = lms[mp_pose.PoseLandmark.RIGHT_HIP]

        if not all(j.visibility > 0.25 for j in [ls, rs, lh, rh]):
            return False

        shoulder_y = (ls.y + rs.y) / 2
        hip_y      = (lh.y + rh.y) / 2

        # Primary: strongly compressed torso (seated rider)
        torso_h = hip_y - shoulder_y
        if torso_h < 0.20:   # v7: tightened from 0.26
            return True

        # Secondary: moderate compression + bent knees
        lk = lms[mp_pose.PoseLandmark.LEFT_KNEE]
        rk = lms[mp_pose.PoseLandmark.RIGHT_KNEE]
        knee_drops = [k.y - hip_y for k in [lk, rk] if k.visibility > 0.25]
        if torso_h < 0.24 and knee_drops and \
                (sum(knee_drops) / len(knee_drops)) < 0.20:
            return True

        # Wide box (rider+bike side view): 1 additional signal needed
        if box_w is not None and box_w > 140:
            lw = lms[mp_pose.PoseLandmark.LEFT_WRIST]
            rw = lms[mp_pose.PoseLandmark.RIGHT_WRIST]
            shoulder_cx = (ls.x + rs.x) / 2
            wrist_near  = sum(
                1 for w in [lw, rw]
                if w.visibility > 0.25 and abs(w.x - shoulder_cx) < 0.30
            ) >= 1
            if torso_h < 0.28 and wrist_near:
                return True

        return False
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════
#  FEATURE EXTRACTORS  (all return 0.0–1.0)
# ═══════════════════════════════════════════════════════════

def feat_orientation(lms, cw):
    """
    v7 RECALIBRATED: side-on is now NEUTRAL (0.50), not a crossing signal.

    Rationale: in dashcam footage almost everyone appears side-on.
    Side orientation alone means nothing — the person might be standing
    and waiting. Only combined with step/motion does it indicate crossing.

    frontal (wide shoulders) → 0.20  (facing/waiting at kerb)
    side-on (narrow)         → 0.50  (neutral — could be waiting or walking)
    back (very narrow)       → 0.35  (ambiguous)
    """
    ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    if not (_vis(ls) and _vis(rs)):
        return 0.35
    sw = abs(ls.x - rs.x)
    if sw > 0.38:  return 0.20   # frontal — clearly waiting/facing camera
    if sw > 0.12:  return 0.50   # side-on — neutral
    return 0.35                   # back to camera — slightly above neutral


def feat_head(lms):
    """Head turn toward road — looking left/right to check traffic."""
    nose = lms[mp_pose.PoseLandmark.NOSE]
    ls   = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs   = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    if not (_vis(nose) and _vis(ls) and _vis(rs)):
        return 0.30
    offset = abs(nose.x - (ls.x + rs.x) / 2)
    return min(0.85, 0.15 + offset * 2.5)


def feat_lean(lms):
    """Lateral lean / body lean forward — step initiation."""
    ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    lh = lms[mp_pose.PoseLandmark.LEFT_HIP]
    rh = lms[mp_pose.PoseLandmark.RIGHT_HIP]
    if not all(_vis(j) for j in [ls, rs, lh, rh]):
        return 0.25
    lean = abs((ls.x + rs.x) / 2 - (lh.x + rh.x) / 2)
    return min(0.85, 0.15 + lean * 3.5)


def feat_step(lms):
    """
    v7 RECALIBRATED: feet-together strongly penalises score.

    Standing still = feet very close together (spread < 0.05) → 0.05
    Mid-stride = significant spread → high score
    This is the strongest NOT CROSSING signal available.
    """
    la = lms[mp_pose.PoseLandmark.LEFT_ANKLE]
    ra = lms[mp_pose.PoseLandmark.RIGHT_ANKLE]
    if not (_vis(la, 0.30) and _vis(ra, 0.30)):
        return 0.28   # unknown — return low default
    spread = abs(la.x - ra.x)
    if spread < 0.05:
        return 0.05   # v7: standing still — strong NOT CROSSING signal
    if spread < 0.10:
        return 0.25   # slight shift — probably still waiting
    return min(0.92, 0.30 + spread * 3.2)


def feat_motion(history, fw, fh):
    """Lateral + approach motion from centroid history."""
    if len(history) < 4:
        return 0.15   # v7: default lowered — no history = assume stationary
    xs = [cx / fw for (cx, cy) in history]
    ys = [cy / fh for (cx, cy) in history]
    dx = abs(xs[-1] - xs[0])
    dy = ys[-1] - ys[0]
    lateral  = min(0.90, 0.10 + dx * 8.0)
    approach = min(0.90, 0.10 + max(dy, 0) * 5.0)
    return max(lateral, approach)


def feat_proximity(x1, x2, fw):
    """Near frame edge = near kerb — slight bias."""
    cx_norm = ((x1 + x2) / 2) / fw
    edge_d  = abs(cx_norm - 0.5)
    return min(0.70, 0.15 + edge_d * 1.0)


# ═══════════════════════════════════════════════════════════
#  INTENT CLASSIFIER
# ═══════════════════════════════════════════════════════════

class IntentClassifier:

    def __init__(self):
        self.tracks      = {}
        self.next_id     = 0
        self._last_boxes = {}

    def _iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        ua    = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / ua if ua > 0 else 0

    def _assign_track(self, boxes):
        new_tracks = {}
        assigned   = set()
        ids_out    = []
        for (x1, y1, x2, y2, conf) in boxes:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            best_id, best_score = None, 0.15
            for tid, lb in self._last_boxes.items():
                if tid in assigned:
                    continue
                iou = self._iou((x1, y1, x2, y2), lb)
                if iou > best_score:
                    best_score, best_id = iou, tid
            if best_id is None:
                for tid, hist in self.tracks.items():
                    if tid in assigned or not hist:
                        continue
                    px, py = hist[-1]
                    if np.hypot(cx - px, cy - py) < 70:
                        best_id = tid
                        break
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            assigned.add(best_id)
            hist = self.tracks.get(best_id, deque(maxlen=HISTORY_LEN))
            hist.append((cx, cy))
            new_tracks[best_id] = hist
            ids_out.append(best_id)
        self.tracks      = new_tracks
        self._last_boxes = {
            tid: (x1, y1, x2, y2)
            for tid, (x1, y1, x2, y2, _) in zip(ids_out, boxes)
        }
        return ids_out

    def classify(self, frame, yolo_boxes, vehicle_boxes=None, segmenter=None):
        fh, fw  = frame.shape[:2]
        vboxes  = vehicle_boxes or []

        ped_boxes = [b for b in yolo_boxes
                     if _is_pedestrian_box(b[0], b[1], b[2], b[3])]

        track_ids = self._assign_track(ped_boxes)
        results   = []

        for tid, (x1, y1, x2, y2, conf) in zip(track_ids, ped_boxes):
            cx      = (x1 + x2) / 2
            cy      = (y1 + y2) / 2
            bw      = x2 - x1
            history = self.tracks.get(tid, deque())

            if _near_vehicle(cx, cy, vboxes):
                continue
            if _on_vehicle(x1, y1, x2, y2, vboxes):
                continue

            # Pose with 2-attempt fallback
            pose_ok, lms, px1, py1, cw, ch = False, None, x1, y1, bw, (y2-y1)
            for pad in [20, 45]:
                _px1 = max(0, x1 - pad);  _py1 = max(0, y1 - pad)
                _px2 = min(fw, x2 + pad); _py2 = min(fh, y2 + pad)
                crop = frame[_py1:_py2, _px1:_px2]
                if crop.shape[0] < 40 or crop.shape[1] < 15:
                    continue
                rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                res  = pose_estimator.process(rgb)
                if res.pose_landmarks is not None:
                    pose_ok = True
                    lms     = res.pose_landmarks.landmark
                    px1, py1 = _px1, _py1
                    ch, cw   = crop.shape[:2]
                    break

            if pose_ok and _is_rider_pose(lms, box_w=bw):
                continue

            # ── Feature extraction ─────────────────────────
            if pose_ok:
                feats = {
                    'orient': feat_orientation(lms, cw),
                    'head'  : feat_head(lms),
                    'lean'  : feat_lean(lms),
                    'step'  : feat_step(lms),
                }
            else:
                feats = {k: 0.22 for k in ['orient', 'head', 'lean', 'step']}

            feats['motion'] = feat_motion(history, fw, fh)
            feats['prox']   = feat_proximity(x1, x2, fw)

            # ── v7 Recalibrated weights ────────────────────
            # motion + step are primary signals (person must be MOVING)
            # orientation is context only
            W = {
                'motion': 0.25,   # v7: most important — must be moving
                'step'  : 0.20,   # v7: feet spread = walking
                'orient': 0.15,   # v7: demoted — side-on is neutral
                'head'  : 0.15,
                'lean'  : 0.13,
                'prox'  : 0.12,
            }
            score = sum(W[k] * feats[k] for k in W)

            # Scene context boost from Module 4
            if segmenter is not None:
                if segmenter.is_in_road(x1, y1, x2, y2):
                    score = min(1.0, score + 0.10)
                elif segmenter.is_near_crosswalk(x1, y1, x2, y2):
                    score = min(1.0, score + 0.05)

            if score >= CROSSING_THRESH:
                label, color = 'CROSSING',  (0,   0, 255)
            elif score >= MAYBE_THRESH:
                label, color = 'MAYBE',     (0, 140, 255)
            else:
                label, color = 'NOT CROSS', (0, 200, 200)

            results.append({
                'box'         : (x1, y1, x2, y2),
                'track_id'    : tid,
                'score'       : score,
                'label'       : label,
                'color'       : color,
                'features'    : feats,
                'pose_ok'     : pose_ok,
                'pose_lms'    : lms,
                'crop_origin' : (px1, py1),
                'crop_size'   : (cw, ch),
            })

        return results


# ═══════════════════════════════════════════════════════════
#  DRAWING
# ═══════════════════════════════════════════════════════════

def draw_intent(display, result):
    x1, y1, x2, y2 = result['box']
    color  = result['color']
    score  = result['score']
    label  = result['label']
    tid    = result['track_id']
    dh, dw = display.shape[:2]

    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

    SHORT = {'NOT CROSS': 'NC', 'CROSSING': 'CRS', 'MAYBE': 'MAY'}
    tag   = f"#{tid} {SHORT[label]} {score:.2f}"
    fs    = 0.42
    (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    tag_y = max(y1, 18)
    cv2.rectangle(display, (x1, tag_y - 18), (x1 + tw + 6, tag_y), color, -1)
    cv2.putText(display, tag, (x1 + 3, tag_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1)

    # Score bar
    bx  = min(x2 + 5, dw - 12)
    bh_ = y2 - y1
    cv2.rectangle(display, (bx, y1), (bx + 7, y2), (40, 40, 40), -1)
    fill = max(y1, min(int(y2 - bh_ * score), y2))
    cv2.rectangle(display, (bx, fill), (bx + 7, y2), color, -1)

    # Pose skeleton
    if result['pose_ok'] and result['pose_lms'] is not None:
        px1_c, py1_c = result['crop_origin']
        cw,    ch    = result['crop_size']
        lms          = result['pose_lms']
        for conn in mp_pose.POSE_CONNECTIONS:
            la, lb = lms[conn[0]], lms[conn[1]]
            if la.visibility < 0.35 or lb.visibility < 0.35:
                continue
            ax  = max(0, min(px1_c + int(la.x * cw), dw - 1))
            ay  = max(0, min(py1_c + int(la.y * ch), dh - 1))
            bx_ = max(0, min(px1_c + int(lb.x * cw), dw - 1))
            by_ = max(0, min(py1_c + int(lb.y * ch), dh - 1))
            cv2.line(display, (ax, ay), (bx_, by_), color, 1)
        for lm in lms:
            if lm.visibility < 0.35:
                continue
            kx = max(0, min(px1_c + int(lm.x * cw), dw - 1))
            ky = max(0, min(py1_c + int(lm.y * ch), dh - 1))
            cv2.circle(display, (kx, ky), 3, (255, 255, 255), -1)


def draw_feature_panel(display, results):
    if not results:
        return
    feat_keys  = ['motion', 'step', 'orient', 'head', 'lean', 'prox']
    feat_short = ['Mot', 'Step', 'Ori', 'Head', 'Lean', 'Prx']
    rows       = min(len(results), 3)
    panel_w    = 225
    row_h      = 52
    panel_h    = 18 + rows * row_h
    dh, dw     = display.shape[:2]
    px = dw - panel_w - 6
    py = max(90, dh - panel_h - 6)

    overlay = display.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, display, 0.28, 0, display)
    cv2.putText(display, 'Intent Features (v7)', (px + 4, py + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1)

    for i, res in enumerate(results[:3]):
        oy = py + 18 + i * row_h
        cv2.putText(display,
                    f"#{res['track_id']} {res['label']} {res['score']:.2f}",
                    (px + 4, oy + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, res['color'], 1)
        bw_bar = 28
        for j, (fk, fs_) in enumerate(zip(feat_keys, feat_short)):
            v   = res['features'].get(fk, 0.22)
            bx  = px + 4 + j * 36
            by_ = oy + 16
            cv2.rectangle(display, (bx, by_), (bx + bw_bar, by_ + 9), (55, 55, 55), -1)
            bc  = (0, 200, 80) if v > 0.60 else (0, 155, 255) if v > 0.35 else (60, 60, 180)
            cv2.rectangle(display, (bx, by_), (bx + int(bw_bar * v), by_ + 9), bc, -1)
            cv2.putText(display, fs_, (bx, by_ + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.23, (140, 140, 140), 1)


# ═══════════════════════════════════════════════════════════
#  STANDALONE WEBCAM DEMO
# ═══════════════════════════════════════════════════════════

def run_webcam():
    print("Loading YOLOv8n...")
    model      = YOLO('yolov8n.pt')
    classifier = IntentClassifier()
    prev_time  = time.time()
    segmenter  = None
    try:
        from module4_segmentation import SceneSegmenter
        segmenter = SceneSegmenter(use_dnn=True, fisheye=True)
        print("Scene segmenter loaded.")
    except Exception as e:
        print(f"Segmenter not loaded: {e}")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    show_seg = segmenter is not None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yolo_res   = model(frame, classes=[0], conf=CONF_THRESHOLD, verbose=False)
        yolo_boxes = [(int(b.xyxy[0][0]), int(b.xyxy[0][1]),
                       int(b.xyxy[0][2]), int(b.xyxy[0][3]), float(b.conf[0]))
                      for r in yolo_res for b in r.boxes]
        veh_res    = model(frame, classes=[1,2,3,5,7], conf=0.28, verbose=False)
        veh_boxes  = [(int(b.xyxy[0][0]), int(b.xyxy[0][1]),
                       int(b.xyxy[0][2]), int(b.xyxy[0][3]))
                      for r in veh_res for b in r.boxes]
        zone_map = segmenter.process(frame) if segmenter else None
        display  = frame.copy()
        if show_seg and zone_map is not None:
            segmenter.draw_overlay(display, zone_map,
                                   person_boxes=[r['box'] for r in
                                   classifier.classify(frame, yolo_boxes,
                                                       veh_boxes, segmenter)])
        intent_results = classifier.classify(frame, yolo_boxes, veh_boxes, segmenter)
        for res in intent_results:
            draw_intent(display, res)
        draw_feature_panel(display, intent_results)
        fps = 1 / (time.time() - prev_time + 0.001)
        prev_time = time.time()
        h, w = display.shape[:2]
        cv2.rectangle(display, (0, 0), (w, 58), (15, 15, 15), -1)
        cv2.putText(display, 'CrossSafe M3 v7 — Intent',
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2)
        cv2.putText(display, f'FPS:{fps:.1f}  Peds:{len(intent_results)}  [Q]Quit',
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        cv2.imshow('CrossSafe M3', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_webcam()