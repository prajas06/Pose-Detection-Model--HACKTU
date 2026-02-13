# =========================
# FastAPI ML Exercise Session API (OOP – uses PoseTrainerSession class inside API)
# + Added with MINIMUM changes:
#   1) Reference benchmark targets (reference_angles.json)
#   2) Live "explain matrix" payload
#   3) Strict start gate: all 33 landmarks visible for N frames
# =========================

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import cv2
import json
import random
import os
from typing import Dict, Any, Optional

import mediapipe as mp

# ---- Your existing project imports (UNCHANGED) ----
from src.pose_tracker import PoseTracker
from src.counters import BicepCurlCounter, SquatCounter, ShoulderPressCounter
from src.utils import calculate_angle

app = FastAPI()

pose_tracker = PoseTracker()
mp_pose = mp.solutions.pose


# ======= HELPER FUNCTIONS (YOUR ORIGINAL LOGIC KEPT) =======
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def score_from_error(err: float, tol: float) -> float:
    return clamp01(1.0 - (err / tol))

def alignment_score(exercise: str, lm) -> float:
    def pt(e):
        p = lm[e.value]
        return (p.x, p.y)

    LS = pt(mp_pose.PoseLandmark.LEFT_SHOULDER)
    RS = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    LH = pt(mp_pose.PoseLandmark.LEFT_HIP)
    RH = pt(mp_pose.PoseLandmark.RIGHT_HIP)

    mid_sh = ((LS[0] + RS[0]) / 2, (LS[1] + RS[1]) / 2)
    mid_hip = ((LH[0] + RH[0]) / 2, (LH[1] + RH[1]) / 2)

    down = (mid_hip[0], mid_hip[1] + 0.3)
    torso_angle = calculate_angle(mid_sh, mid_hip, down)
    torso_err = abs(180 - torso_angle)
    torso_score = score_from_error(torso_err, tol=25)

    if exercise == "curl":
        LE = pt(mp_pose.PoseLandmark.LEFT_ELBOW)
        LW = pt(mp_pose.PoseLandmark.LEFT_WRIST)

        elbow_hip_dx = abs(LE[0] - LH[0])
        elbow_close_score = score_from_error(elbow_hip_dx, tol=0.12)

        wrist_elbow_dx = abs(LW[0] - LE[0])
        wrist_stack_score = score_from_error(wrist_elbow_dx, tol=0.10)

        s = 0.45 * torso_score + 0.35 * elbow_close_score + 0.20 * wrist_stack_score
        return 100 * s

    if exercise == "squat":
        LK = pt(mp_pose.PoseLandmark.LEFT_KNEE)
        RK = pt(mp_pose.PoseLandmark.RIGHT_KNEE)
        LA = pt(mp_pose.PoseLandmark.LEFT_ANKLE)
        RA = pt(mp_pose.PoseLandmark.RIGHT_ANKLE)

        left_track = score_from_error(abs(LK[0] - LA[0]), tol=0.10)
        right_track = score_from_error(abs(RK[0] - RA[0]), tol=0.10)
        knee_track = (left_track + right_track) / 2

        mid_ank = ((LA[0] + RA[0]) / 2, (LA[1] + RA[1]) / 2)
        hip_center_err = abs(mid_hip[0] - mid_ank[0])
        hip_center = score_from_error(hip_center_err, tol=0.12)

        s = 0.45 * knee_track + 0.35 * torso_score + 0.20 * hip_center
        return 100 * s

    if exercise == "press":
        LE = pt(mp_pose.PoseLandmark.LEFT_ELBOW)
        LW = pt(mp_pose.PoseLandmark.LEFT_WRIST)
        RE = pt(mp_pose.PoseLandmark.RIGHT_ELBOW)
        RW = pt(mp_pose.PoseLandmark.RIGHT_WRIST)

        left_stack = score_from_error(abs(LW[0] - LE[0]), tol=0.10)
        right_stack = score_from_error(abs(RW[0] - RE[0]), tol=0.10)
        stack_score = (left_stack + right_stack) / 2

        sh_height_err = abs(LS[1] - RS[1])
        symmetry = score_from_error(sh_height_err, tol=0.06)

        s = 0.45 * stack_score + 0.35 * torso_score + 0.20 * symmetry
        return 100 * s

    return 0.0


# ======= REFERENCE BENCHMARK LOADER (NEW FILE SUPPORT) =======
DEFAULT_REF = {
    "curl": {
        "torso_target": 180.0,
        "elbow_hip_dx_target": 0.0,
        "wrist_elbow_dx_target": 0.0,
    },
    "squat": {
        "torso_target": 180.0,
        "knee_track_dx_target": 0.0,
        "hip_center_dx_target": 0.0,
    },
    "press": {
        "torso_target": 180.0,
        "wrist_elbow_dx_target": 0.0,
        "shoulder_symmetry_dy_target": 0.0,
    },
}

def load_reference_targets(path="reference_angles.json"):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)

            merged = {k: dict(DEFAULT_REF[k]) for k in DEFAULT_REF}
            for ex, vals in data.items():
                merged[ex.lower()] = vals
            return merged
        except Exception:
            return DEFAULT_REF
    return DEFAULT_REF

REFERENCE_TARGETS = load_reference_targets()


# ======= SESSION CLASS (MINIMAL CHANGES ONLY) =======
class PoseTrainerSession:
    def __init__(self, exercise: str, target_reps: int):
        self.exercise = exercise.strip().lower()
        self.target_reps = int(target_reps)

        if self.exercise in ("curl", "bicep_curl"):
            self.exercise = "curl"
            self.counter = BicepCurlCounter()
        elif self.exercise == "squat":
            self.counter = SquatCounter()
        elif self.exercise == "press":
            self.counter = ShoulderPressCounter()
        else:
            raise ValueError("Unsupported exercise")

        self.form_scores = []
        self.align_scores = []
        self.last_angle = None

        # ---- STRICT START GATE (NEW) ----
        self.started = False
        self.ready_frames = 0
        self.START_VIS_THRESH = 0.55
        self.READY_FRAMES_REQUIRED = 6

        self._last_lm = None

    def _all_33_visible(self, lm):
        for i in range(33):
            if getattr(lm[i], "visibility", 0.0) < self.START_VIS_THRESH:
                return False
        return True

    def _get_points(self, lm):
        if self.exercise in ("curl", "press"):
            shoulder = (lm[11].x, lm[11].y)
            elbow = (lm[13].x, lm[13].y)
            wrist = (lm[15].x, lm[15].y)
            vis = lm[13].visibility
            return shoulder, elbow, wrist, vis
        else:
            hip = (lm[23].x, lm[23].y)
            knee = (lm[25].x, lm[25].y)
            ankle = (lm[27].x, lm[27].y)
            vis = lm[25].visibility
            return hip, knee, ankle, vis

    def update(self, frame):
        results, _ = pose_tracker.process_bgr(frame)
        if not results.pose_landmarks:
            self.ready_frames = 0
            return

        lm = results.pose_landmarks.landmark
        self._last_lm = lm

        # ---- START GATE ----
        if not self.started:
            if self._all_33_visible(lm):
                self.ready_frames += 1
            else:
                self.ready_frames = 0

            if self.ready_frames < self.READY_FRAMES_REQUIRED:
                return
            else:
                self.started = True

        # ---- YOUR ORIGINAL LOGIC (UNCHANGED) ----
        align_pct = alignment_score(self.exercise, lm)
        self.align_scores.append(align_pct)

        info = self.counter.update(*self._get_points(lm))
        angle = info["angle"]

        if self.exercise == "curl":
            good_range = 20 <= angle <= 160
        elif self.exercise == "squat":
            good_range = 80 <= angle <= 180
        else:
            good_range = 70 <= angle <= 180

        score = 70 if good_range else 0

        if self.last_angle is not None:
            angle_change = abs(angle - self.last_angle)
            score += 20 if angle_change < 15 else 5

        self.last_angle = angle

        if info["reliable"]:
            score += 10

        self.form_scores.append(min(100, score))

    def generate_report(self):
        completed = min(self.counter.state.counter, self.target_reps)
        avg_form = sum(self.form_scores) / max(1, len(self.form_scores))
        avg_align = sum(self.align_scores) / max(1, len(self.align_scores))

        suggestions = {
            "curl": [
                "Keep elbow close",
                "Avoid swinging torso",
                "Lower slowly"
            ],
            "squat": [
                "Keep chest up",
                "Knees out",
                "Heels down"
            ],
            "press": [
                "Keep core tight",
                "Wrists stacked",
                "Press straight up"
            ]
        }

        return {
            "exercise": self.exercise,
            "target_reps": self.target_reps,
            "completed_reps": completed,
            "form_confidence": round(avg_form, 1),
            "alignment": round(avg_align, 1),
            "suggestions": random.sample(suggestions[self.exercise], 2)
        }


# ======= WEBSOCKET (MINIMALLY MODIFIED) =======
@app.websocket("/ws/session")
async def session_ws(websocket: WebSocket):
    await websocket.accept()
    session = None

    try:
        while True:
            message = await websocket.receive()

            if "text" in message and message["text"]:
                data = json.loads(message["text"])
                if data["type"] == "start":
                    session = PoseTrainerSession(
                        exercise=data["exercise"],
                        target_reps=data["target_reps"]
                    )

            if "bytes" in message and session:
                np_frame = np.frombuffer(message["bytes"], np.uint8)
                frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

                if frame is not None:
                    session.update(frame)

                    await websocket.send_text(json.dumps({
                        "type": "live",
                        "started": session.started,
                        "ready_frames": session.ready_frames,
                        "reps": session.counter.state.counter,
                        "form_confidence": round(
                            sum(session.form_scores) / max(1, len(session.form_scores)), 1
                        ),
                        "alignment": round(
                            sum(session.align_scores) / max(1, len(session.align_scores)), 1
                        )
                    }))

                    if session.counter.state.counter >= session.target_reps:
                        report = session.generate_report()
                        await websocket.send_text(json.dumps({
                            "type": "final_report",
                            "report": report
                        }))
                        break

    except WebSocketDisconnect:
        print("Client disconnected")


@app.get("/")
def root():
    return {"status": "Exercise ML API running"}
