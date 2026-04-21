import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import math

# ====================== SORT 跟踪器 ======================
class Track:
    def __init__(self, prediction, track_id, track_lifetime):
        self.prediction = np.array(prediction).reshape(4, 1)
        self.track_id = track_id
        self.track_lifetime = track_lifetime
        self.age = 0
        self.total_visible_count = 1
        self.consecutive_invisible_count = 0

    def predict(self, kf):
        kf_copy = KalmanFilter(dim_x=4, dim_z=2)
        kf_copy.F = kf.F.copy()
        kf_copy.H = kf.H.copy()
        kf_copy.R = kf.R.copy()
        kf_copy.Q = kf.Q.copy()
        kf_copy.P = kf.P.copy()
        kf_copy.x = self.prediction.copy()
        kf_copy.predict()
        self.prediction = kf_copy.x
        self.age += 1

    def update(self, detection, kf):
        kf_copy = KalmanFilter(dim_x=4, dim_z=2)
        kf_copy.F = kf.F.copy()
        kf_copy.H = kf.H.copy()
        kf_copy.R = kf.R.copy()
        kf_copy.Q = kf.Q.copy()
        kf_copy.P = kf.P.copy()
        kf_copy.x = self.prediction.copy()
        kf_copy.update(detection.reshape(2, 1))
        self.prediction = kf_copy.x
        self.total_visible_count += 1
        self.consecutive_invisible_count = 0

    def mark_missed(self):
        self.consecutive_invisible_count += 1

    def is_dead(self):
        return self.consecutive_invisible_count >= self.track_lifetime


class SORTTracker:
    def __init__(self, track_lifetime=20):
        self.next_track_id = 0
        self.tracks = []
        self.track_lifetime = track_lifetime

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.kf.R = np.eye(2)
        self.kf.Q = np.eye(4) * 0.01
        self.kf.P = np.eye(4) * 1000

    def update(self, detections):
        centroids = []
        for rect in detections:
            cx = (rect[0] + rect[2]) / 2
            cy = (rect[1] + rect[3]) / 2
            centroids.append(np.array([cx, cy]))
        centroids = np.array(centroids)

        for track in self.tracks:
            track.predict(self.kf)

        if len(self.tracks) > 0 and len(centroids) > 0:
            cost_matrix = np.zeros((len(self.tracks), len(centroids)))

            for i, track in enumerate(self.tracks):
                for j, det in enumerate(centroids):
                    cost_matrix[i, j] = np.linalg.norm(
                        track.prediction[:2].flatten() - det
                    )

            rows, cols = linear_sum_assignment(cost_matrix)

            matched_tracks = set()
            matched_dets = set()

            for r, c in zip(rows, cols):
                if cost_matrix[r, c] < 100:
                    self.tracks[r].update(centroids[c], self.kf)
                    matched_tracks.add(r)
                    matched_dets.add(c)

            for i in range(len(self.tracks)):
                if i not in matched_tracks:
                    self.tracks[i].mark_missed()

            for j in range(len(centroids)):
                if j not in matched_dets:
                    new_track = Track([centroids[j][0], centroids[j][1], 0, 0],
                                      self.next_track_id, self.track_lifetime)
                    self.tracks.append(new_track)
                    self.next_track_id += 1

        elif len(centroids) > 0:
            for c in centroids:
                self.tracks.append(
                    Track([c[0], c[1], 0, 0], self.next_track_id, self.track_lifetime)
                )
                self.next_track_id += 1

        else:
            for t in self.tracks:
                t.mark_missed()

        self.tracks = [t for t in self.tracks if not t.is_dead()]

        objects = {}
        for t in self.tracks:
            objects[t.track_id] = t.prediction[:2].flatten()

        return objects


# ====================== 角度计算 ======================
def get_target_angles(x_t, y_t, W, H, h_fov_deg, v_fov_deg):
    h_fov_rad = math.radians(h_fov_deg)
    v_fov_rad = math.radians(v_fov_deg)

    f_p_h = W / (2 * math.tan(h_fov_rad / 2))
    f_p_v = H / (2 * math.tan(v_fov_rad / 2))

    dx = x_t - (W / 2)
    dy = y_t - (H / 2)

    yaw = math.degrees(math.atan(dx / f_p_h))
    pitch = math.degrees(math.atan(dy / f_p_v))

    return yaw, pitch


# ====================== 主程序 ======================
if __name__ == "__main__":

    print("Loading model...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="E:/vscode_local/inference_single/yolo26.pt",
        confidence_threshold=0.5,
        device="cuda:0"  # 不行就改cpu
    )

    cap = cv2.VideoCapture("E:/vscode_local/inference_single/real_test_video.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        "E:/vscode_local/inference_single/output3.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    tracker = SORTTracker()

    # ===== 角度历史（关键）=====
    angle_history = {}

    dt = 1.0 / fps
    alpha = 0.7  # 平滑系数

    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        print(f"Processing frame {frame_id}")

        result = get_sliced_prediction(
            frame,
            detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )

        rects = []
        for pred in result.object_prediction_list:
            box = pred.bbox.to_xyxy()
            rects.append(box)

            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            angle_x, angle_y = get_target_angles(cx, cy, w, h, 120, 90)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{pred.category.name} {pred.score.value:.2f}",
                        (x1, y1-20), 0, 0.5, (0,255,0), 2)
            cv2.putText(frame, f"X:{angle_x:.1f} Y:{angle_y:.1f}",
                        (x1, y1-5), 0, 0.5, (255,255,0), 2)

        objects = tracker.update(rects)

        for obj_id, (cx, cy) in objects.items():
            cx, cy = int(cx), int(cy)

            yaw, pitch = get_target_angles(cx, cy, w, h, 120, 90)

            yaw_rate, pitch_rate = 0.0, 0.0

            if obj_id in angle_history:
                prev = angle_history[obj_id]

                raw_yaw_rate = (yaw - prev["yaw"]) / dt
                raw_pitch_rate = (pitch - prev["pitch"]) / dt

                # ===== 平滑 =====
                yaw_rate = alpha * raw_yaw_rate + (1 - alpha) * prev.get("yaw_rate", raw_yaw_rate)
                pitch_rate = alpha * raw_pitch_rate + (1 - alpha) * prev.get("pitch_rate", raw_pitch_rate)

            angle_history[obj_id] = {
                "yaw": yaw,
                "pitch": pitch,
                "yaw_rate": yaw_rate,
                "pitch_rate": pitch_rate
            }

            cv2.putText(frame, f"ID:{obj_id}", (cx, cy),
                        0, 0.7, (0,0,255), 2)
            cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

            cv2.putText(frame,
                f"Vy:{yaw_rate:.1f} Vp:{pitch_rate:.1f}",
                (cx, cy + 20),
                0, 0.5, (0,255,255), 2)

        out.write(frame)

    cap.release()
    out.release()

    print("✅ 完成：角度 + 角速度 已输出视频")
