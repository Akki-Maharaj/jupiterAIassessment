"""
Multimodal Robotics Data Pipeline: Core OpenCV Implementation
Assessment: Jupiter AI Labs - ML Engineer (Robotics)

Core Tasks Implemented:
1. Video Inspection & Metadata Extraction (Tasks 1 & 4)
2. Multi-Camera Synchronization & 2x2 Tiled Video (Task 2)
3. 21-Point Hand-Keypoint Annotation (Task 3)
4. Visual Odometry / SLAM Camera Trajectory Estimation (Task 5)
"""

import os
import json
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Setup input and output directories
EGO_DIR = Path("drive-download-20261015T111339Z-1-001/ego")
UMI_DIR = Path("drive-download-20261015T111339Z-1-001/umi")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# -------------------------------------------------------------------------
# Task 1: Video Inspection & Preprocessing (OpenCV)
# -------------------------------------------------------------------------
def inspect_videos():
    print("\n--- Task 1: Video Inspection ---")
    def get_video_info(video_path):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frames / fps if fps > 0 else 0
        cap.release()
        return {
            "file": video_path.name,
            "resolution": f"{width}x{height}",
            "fps": round(fps, 2),
            "frames": frames,
            "duration_s": round(duration, 2)
        }

    video_files = sorted(list(EGO_DIR.glob("*.MP4"))) + sorted(list(UMI_DIR.glob("*.MP4")))
    print(f"{'File':<25} | {'Resolution':<12} | {'FPS':<6} | {'Frames':<8} | {'Duration (s)':<12}")
    print("-" * 75)
    for vf in video_files:
        info = get_video_info(vf)
        print(f"{info['file']:<25} | {info['resolution']:<12} | {info['fps']:<6} | {info['frames']:<8} | {info['duration_s']:<12}")


# -------------------------------------------------------------------------
# Task 2: Ego Camera Multi-View Synchronization (2x2 Grid)
# -------------------------------------------------------------------------
def create_synchronized_2x2(max_frames=150):
    print("\n--- Task 2: Synchronizing Ego Cameras (2x2 Tiled Video) ---")
    ego_paths = sorted(list(EGO_DIR.glob("*.MP4")))
    caps = [cv2.VideoCapture(str(p)) for p in ego_paths]
    names = [p.stem.split('_')[0] for p in ego_paths]

    fps = caps[0].get(cv2.CAP_PROP_FPS)
    sync_output_path = OUTPUT_DIR / "ego_synchronized_2x2.mp4"
    writer = cv2.VideoWriter(str(sync_output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))

    for _ in range(max_frames):
        frames = []
        for cap, name in zip(caps, names):
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (640, 360))
            cv2.putText(resized, f"Cam: {name}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            frames.append(resized)
        
        if len(frames) != 4:
            break
            
        top_row = np.hstack([frames[0], frames[1]])
        bottom_row = np.hstack([frames[2], frames[3]])
        tiled_frame = np.vstack([top_row, bottom_row])
        writer.write(tiled_frame)

    for cap in caps:
        cap.release()
    writer.release()
    print(f"Synchronized 2x2 video saved: {sync_output_path}")


# -------------------------------------------------------------------------
# Task 3: Ego 21-Point Hand-Keypoint Annotation
# -------------------------------------------------------------------------
def annotate_hand_keypoints(max_frames=100):
    print("\n--- Task 3: Hand-Keypoint Detection & Annotation ---")
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
        (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
        (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Pinky & Palm
    ]

    # Ensure model file exists
    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        print("Downloading hand_landmarker.task model...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    video_path = EGO_DIR / "HL_CH01_005_1013.MP4"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    annotated_video_path = OUTPUT_DIR / "annotated_hands.mp4"
    writer = cv2.VideoWriter(str(annotated_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    annotations = []
    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect(mp_img)
        
        frame_data = {"frame_idx": frame_idx, "hands": []}
        if result.hand_landmarks:
            for hand_idx, landmarks in enumerate(result.hand_landmarks):
                handedness = result.handedness[hand_idx][0].category_name if result.handedness else "Hand"
                score = float(result.handedness[hand_idx][0].score) if result.handedness else 1.0
                
                pts = []
                for lm in landmarks:
                    pts.append({"x": round(lm.x, 4), "y": round(lm.y, 4), "z": round(lm.z, 4)})
                    px, py = int(lm.x * W), int(lm.y * H)
                    cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
                
                for start_idx, end_idx in HAND_CONNECTIONS:
                    p1 = (int(landmarks[start_idx].x * W), int(landmarks[start_idx].y * H))
                    p2 = (int(landmarks[end_idx].x * W), int(landmarks[end_idx].y * H))
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)
                    
                frame_data["hands"].append({"label": handedness, "score": score, "landmarks": pts})
                
        annotations.append(frame_data)
        writer.write(frame)

    cap.release()
    writer.release()

    json_path = OUTPUT_DIR / "hand_keypoints.json"
    with open(json_path, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"Annotated hand video saved: {annotated_video_path}")
    print(f"Keypoints saved: {json_path}")


# -------------------------------------------------------------------------
# Task 4: Visual Odometry / SLAM Camera Trajectory Estimation
# -------------------------------------------------------------------------
def estimate_camera_trajectory(max_frames=150):
    print("\n--- Task 4: Visual Odometry Trajectory Estimation ---")
    video_path = EGO_DIR / "HL_CH01_005_1013.MP4"
    cap = cv2.VideoCapture(str(video_path))

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Approximate pinhole intrinsic matrix K
    K = np.array([
        [1000.0,    0.0, W / 2.0],
        [   0.0, 1000.0, H / 2.0],
        [   0.0,    0.0,     1.0]
    ], dtype=np.float64)

    cur_R = np.eye(3)
    cur_t = np.zeros((3, 1))
    trajectory = [cur_t.flatten()]

    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=15)

    for _ in range(max_frames):
        ret, curr_frame = cap.read()
        if not ret:
            break
            
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Lucas-Kanade optical flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]
        
        if len(good_prev) >= 8:
            E, _ = cv2.findEssentialMat(good_curr, good_prev, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, _ = cv2.recoverPose(E, good_curr, good_prev, K)
            
            cur_t = cur_t + cur_R.dot(t)
            cur_R = cur_R.dot(R)
            trajectory.append(cur_t.flatten())
        
        if len(good_curr) < 200:
            prev_pts = cv2.goodFeaturesToTrack(curr_gray, maxCorners=500, qualityLevel=0.01, minDistance=15)
        else:
            prev_pts = good_curr.reshape(-1, 1, 2)
            
        prev_gray = curr_gray

    cap.release()
    trajectory = np.array(trajectory)

    traj_csv = OUTPUT_DIR / "camera_trajectory.csv"
    np.savetxt(traj_csv, trajectory, delimiter=",", header="x,y,z", comments="")
    print(f"Saved camera trajectory ({len(trajectory)} poses) to: {traj_csv}")


if __name__ == "__main__":
    inspect_videos()
    create_synchronized_2x2(max_frames=150)
    annotate_hand_keypoints(max_frames=100)
    estimate_camera_trajectory(max_frames=150)
    print("\nAll Core OpenCV Tasks Completed Successfully!")
