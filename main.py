import sys
from pathlib import Path
import cv2

# Keep your exact project-root logic
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(PROJECT_ROOT))

from src.config import PoseConfig, CurlConfig, SquatConfig, ShoulderPressConfig
from src.pose_tracker import PoseTracker
from src.counters import BicepCurlCounter, SquatCounter, ShoulderPressCounter
from src.utils import calculate_angle


class PoseTrainerApp:
    # =========================
    # POSE TRAINER (curl/squat/press)
    # Start gate runs ONLY ONCE
    # Shows Form Confidence + Alignment
    # Final report includes suggestions
    # FIXED: always uses waitKey to prevent "Not Responding"
    # =========================

    def __init__(self):
        # --- configs (same as your code) ---
        self.pose_cfg = PoseConfig()
        self.curl_cfg = CurlConfig()
        self.squat_cfg = SquatConfig()
        self.press_cfg = ShoulderPressConfig()

        # --- tracker (same as your code) ---
        self.tracker = PoseTracker(
            min_det_conf=self.pose_cfg.min_detection_confidence,
            min_track_conf=self.pose_cfg.min_tracking_confidence
        )

        # --- mediapipe pose alias (same as your code) ---
        self.mp_pose = __import__("mediapipe").solutions.pose

        # --- user selections filled later ---
        self.choice = None
        self.TARGET_REPS = None
        self.EXERCISE_NAME = None
        self.counter = None

        # ---------- START CONDITION (same constants) ----------
        self.START_VIS_THRESH = 0.55
        self.READY_FRAMES_REQUIRED = 6
        self.started = False
        self.ready_frames = 0

        # ---------- TRACKING (same vars) ----------
        self.form_scores = []
        self.align_scores = []
        self.last_angle = None

    # ----- helpers: exact same math/behavior -----
    def clamp01(self, x):
        return max(0.0, min(1.0, x))

    def score_from_error(self, err, tol):
        return self.clamp01(1.0 - (err / tol))

    def alignment_score(self, choice, lm):
        def pt(e):
            p = lm[e.value]
            return (p.x, p.y)

        LS = pt(self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        RS = pt(self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        LH = pt(self.mp_pose.PoseLandmark.LEFT_HIP)
        RH = pt(self.mp_pose.PoseLandmark.RIGHT_HIP)

        mid_sh = ((LS[0] + RS[0]) / 2, (LS[1] + RS[1]) / 2)
        mid_hip = ((LH[0] + RH[0]) / 2, (LH[1] + RH[1]) / 2)

        down = (mid_hip[0], mid_hip[1] + 0.3)
        torso_angle = calculate_angle(mid_sh, mid_hip, down)
        torso_err = abs(180 - torso_angle)
        torso_score = self.score_from_error(torso_err, tol=25)

        if choice == "curl":
            LE = pt(self.mp_pose.PoseLandmark.LEFT_ELBOW)
            LW = pt(self.mp_pose.PoseLandmark.LEFT_WRIST)

            elbow_hip_dx = abs(LE[0] - LH[0])
            elbow_close_score = self.score_from_error(elbow_hip_dx, tol=0.12)

            wrist_elbow_dx = abs(LW[0] - LE[0])
            wrist_stack_score = self.score_from_error(wrist_elbow_dx, tol=0.10)

            s = 0.45 * torso_score + 0.35 * elbow_close_score + 0.20 * wrist_stack_score
            return 100 * s

        if choice == "squat":
            LK = pt(self.mp_pose.PoseLandmark.LEFT_KNEE)
            RK = pt(self.mp_pose.PoseLandmark.RIGHT_KNEE)
            LA = pt(self.mp_pose.PoseLandmark.LEFT_ANKLE)
            RA = pt(self.mp_pose.PoseLandmark.RIGHT_ANKLE)

            left_track = self.score_from_error(abs(LK[0] - LA[0]), tol=0.10)
            right_track = self.score_from_error(abs(RK[0] - RA[0]), tol=0.10)
            knee_track = (left_track + right_track) / 2

            mid_ank = ((LA[0] + RA[0]) / 2, (LA[1] + RA[1]) / 2)
            hip_center_err = abs(mid_hip[0] - mid_ank[0])
            hip_center = self.score_from_error(hip_center_err, tol=0.12)

            s = 0.45 * knee_track + 0.35 * torso_score + 0.20 * hip_center
            return 100 * s

        if choice == "press":
            LE = pt(self.mp_pose.PoseLandmark.LEFT_ELBOW)
            LW = pt(self.mp_pose.PoseLandmark.LEFT_WRIST)
            RE = pt(self.mp_pose.PoseLandmark.RIGHT_ELBOW)
            RW = pt(self.mp_pose.PoseLandmark.RIGHT_WRIST)

            left_stack = self.score_from_error(abs(LW[0] - LE[0]), tol=0.10)
            right_stack = self.score_from_error(abs(RW[0] - RE[0]), tol=0.10)
            stack_score = (left_stack + right_stack) / 2

            sh_height_err = abs(LS[1] - RS[1])
            symmetry = self.score_from_error(sh_height_err, tol=0.06)

            s = 0.45 * stack_score + 0.35 * torso_score + 0.20 * symmetry
            return 100 * s

        return 0.0

    # ----- user input & counter selection: same branching -----
    def get_user_input_and_setup(self):
        self.choice = input("Enter exercise (curl/squat/press): ").strip().lower()
        self.TARGET_REPS = int(input("Enter target reps: ").strip())

        if self.choice == "curl":
            self.EXERCISE_NAME = "Bicep Curl (Left Arm)"
            self.counter = BicepCurlCounter(
                up_angle=self.curl_cfg.up_angle,
                down_angle=self.curl_cfg.down_angle,
                visibility_threshold=self.curl_cfg.visibility_threshold
            )
        elif self.choice == "squat":
            self.EXERCISE_NAME = "Squat"
            self.counter = SquatCounter(
                up_angle=self.squat_cfg.up_angle,
                down_angle=self.squat_cfg.down_angle,
                visibility_threshold=self.squat_cfg.visibility_threshold
            )
        elif self.choice == "press":
            self.EXERCISE_NAME = "Shoulder Press"
            self.counter = ShoulderPressCounter(
                up_angle=self.press_cfg.up_angle,
                down_angle=self.press_cfg.down_angle,
                visibility_threshold=self.press_cfg.visibility_threshold
            )
        else:
            raise ValueError("Invalid input. Type only: curl, squat, press")

    # ----- render header: same text/positions -----
    def draw_header(self, annotated):
        cv2.putText(annotated, f"Exercise: {self.EXERCISE_NAME}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated, f"Target: {self.TARGET_REPS}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # ----- start gate: same steps, same continues/breaks -----
    def handle_start_gate(self, results, annotated):
        if results.pose_landmarks is None:
            self.ready_frames = 0
            cv2.putText(annotated, "POSE NOT DETECTED - step back, full body in frame", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            cv2.imshow("Pose Trainer", annotated)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                return "break"
            return "continue"

        lm = results.pose_landmarks.landmark

        KEY_LMS = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_ANKLE,
        ]

        key_visible = all(
            getattr(lm[e.value], "visibility", 0.0) >= self.START_VIS_THRESH
            for e in KEY_LMS
        )

        self.ready_frames = (self.ready_frames + 1) if key_visible else 0

        cv2.putText(annotated, "Stand ready in full frame", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, f"Getting ready: {self.ready_frames}/{self.READY_FRAMES_REQUIRED}",
                    (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if self.ready_frames >= self.READY_FRAMES_REQUIRED:
            self.started = True
            cv2.putText(annotated, "START ✅", (10, 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            cv2.imshow("Pose Trainer", annotated)
            cv2.waitKey(600)  # short pause so user sees START
            return "started"
        else:
            cv2.imshow("Pose Trainer", annotated)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                return "break"
            return "continue"

    # ----- after start: pose lost handling exactly same -----
    def handle_pose_lost_after_start(self, results, annotated):
        if results.pose_landmarks is None:
            cv2.putText(annotated, "Pose lost - stay in frame", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow("Pose Trainer", annotated)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                return "break"
            return "continue"
        return "ok"

    # ----- main per-frame logic: rep counting + scoring unchanged -----
    def process_frame_after_start(self, results, annotated):
        lm = results.pose_landmarks.landmark

        # Alignment (live)
        align_pct = self.alignment_score(self.choice, lm)
        self.align_scores.append(align_pct)
        avg_align_live = sum(self.align_scores) / max(1, len(self.align_scores))

        # Rep counting
        if self.choice in ("curl", "press"):
            shoulder = (lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
            elbow = (lm[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     lm[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y)
            wrist = (lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y)

            vis = lm[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility
            info = self.counter.update(shoulder, elbow, wrist, vis)
        else:
            hip = (lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].y)
            knee = (lm[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    lm[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y)
            ankle = (lm[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     lm[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y)

            vis = lm[self.mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
            info = self.counter.update(hip, knee, ankle, vis)

        # Form confidence (performance)
        angle = info["angle"]
        if self.choice == "curl":
            good_range = 20 <= angle <= 160
        elif self.choice == "squat":
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
        avg_form_live = sum(self.form_scores) / max(1, len(self.form_scores))

        # Display
        cv2.putText(annotated, f"Angle: {angle:.1f}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated, f"Reps: {info['counter']} / {self.TARGET_REPS}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated, f"Form Confidence: {avg_form_live:.1f}%", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated, f"Alignment: {avg_align_live:.1f}%", (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Pose Trainer", annotated)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            return "break"

        if info["counter"] >= self.TARGET_REPS:
            return "done"

        return "continue"

    # ----- final report: identical print lines + suggestions -----
    def print_final_report(self):
        completed_reps = self.counter.state.counter
        avg_form = sum(self.form_scores) / max(1, len(self.form_scores))
        avg_align = sum(self.align_scores) / max(1, len(self.align_scores))

        print("\n===== PERFORMANCE REPORT =====")
        print(f"Exercise        : {self.EXERCISE_NAME}")
        print(f"Target Reps     : {self.TARGET_REPS}")
        print(f"Completed Reps  : {completed_reps}")
        print(f"Form Confidence : {avg_form:.1f}%")
        print(f"Alignment       : {avg_align:.1f}%")

        print("\n--- SUGGESTIONS ---")
        if self.choice == "curl":
            print("• Keep your elbow close to your body (don’t let it drift forward).")
            print("• Avoid swinging your torso — stay upright.")
            print("• Control the lowering phase (don’t drop quickly).")
            print("• Keep wrist neutral (in line with forearm).")
        elif self.choice == "squat":
            print("• Keep chest up and spine neutral (avoid rounding).")
            print("• Knees track outward (avoid caving inward).")
            print("• Keep heels down and weight mid-foot.")
            print("• Go as deep as comfortable with control.")
        elif self.choice == "press":
            print("• Keep core tight; don’t lean back.")
            print("• Wrists stacked over elbows at the bottom.")
            print("• Press straight overhead, not forward.")
            print("• Keep shoulders level (avoid tilting).")

        print("================================\n")

    # ----- run: same loop structure, same breaks/continues -----
    def run(self):
        self.get_user_input_and_setup()

        cap = cv2.VideoCapture(self.pose_cfg.camera_index)
        if not cap.isOpened():
            raise RuntimeError("Camera could not be opened. Try changing PoseConfig.camera_index to 0 or 1.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results, annotated = self.tracker.process_bgr(frame)

            # Always draw header
            self.draw_header(annotated)

            # ---------- START GATE (ONLY UNTIL STARTED) ----------
            if not self.started:
                status = self.handle_start_gate(results, annotated)
                if status == "break":
                    break
                if status == "continue":
                    continue
                # status == "started" falls through to after-start behavior in next loop iteration
                # (same as your original code: you show START, then proceed)

            # ---------- AFTER START ----------
            pose_ok = self.handle_pose_lost_after_start(results, annotated)
            if pose_ok == "break":
                break
            if pose_ok == "continue":
                continue

            status = self.process_frame_after_start(results, annotated)
            if status == "break":
                break
            if status == "done":
                break
            # else continue loop

        cap.release()
        cv2.destroyAllWindows()

        # Final report
        self.print_final_report()


if __name__ == "__main__":
    app = PoseTrainerApp()
    app.run()
