# Multimodal Robotics Data Pipeline: Core Computer Vision Implementation

**Role Assessment**: ML Engineer – Robotics (Jupiter AI Labs)  
**Author**: Akshat
**Implementation**: Python, OpenCV, MediaPipe, NumPy, Matplotlib  

---

## 1. Project Overview

This repository implements the core computer vision data-processing pipeline for the Multimodal Robotics Assessment. Given raw recordings from four egocentric head/wrist cameras (`ego/`) and four Universal Manipulation Interface cameras (`umi/`), this pipeline accomplishes:

1. **Eight-Video Inspection & Technical Profiling** (Tasks 1 & 4): Inspects duration, FPS, resolution, codec, and frame counts.
2. **Ego Camera Temporal Alignment & 2x2 Tiled Video** (Task 2): Synchronizes and stacks the 4 ego streams (`HL`, `HR`, `WL`, `WR`) into a single composite $1280 \times 720$ video.
3. **21-Point Hand-Keypoint Annotation** (Task 3): Extracts 21 3D landmarks per hand frame-by-frame using MediaPipe and OpenCV, rendering skeleton previews and saving structured JSON annotations.
4. **Monocular Visual Odometry & Trajectory Estimation** (Task 5): Extracts camera translation and rotation using Shi-Tomasi feature detection, Lucas-Kanade optical flow, Essential Matrix decomposition, and pose recovery.

---

## 2. Environment Setup & Hardware

### System Requirements
* **Operating System**: Windows 11 / Linux / macOS
* **Python Version**: Python 3.12 (tested on Python 3.12.10)
* **Hardware Used**: Standard x86_64 CPU (no external GPU required for core CV pipeline)

### Pinned Dependencies (`requirements.txt`)
* `opencv-contrib-python>=4.10.0`
* `mediapipe>=0.10.14`
* `numpy>=1.26.0`
* `matplotlib>=3.8.0`
* `protobuf>=4.25.3`

*Note on LeRobot*: If converting to LeRobot format (Task 6), pin `lerobot==0.2.1`.

---

## 3. Execution Commands

### Option A: Interactive Jupyter Notebook (Recommended)
Launch the self-contained notebook:
```bash
jupyter notebook opencv_robotics_pipeline.ipynb
```
Run all cells sequentially. The notebook generates inline plots, preview frames, and all output artifacts.

### Option B: Standalone Python Script
Execute the entire pipeline in non-interactive batch mode:
```bash
python opencv_robotics_pipeline.py
```

---

## 4. Expected Outputs

All processed artifacts are written to the [`output/`](./output) directory:

| Output File | Description | Size / Scope |
| :--- | :--- | :--- |
| `output/ego_synchronized_2x2.mp4` | Synchronized 2x2 grid video of Head & Wrist cameras | $1280 \times 720$ @ 29.97 FPS |
| `output/annotated_hands.mp4` | Egocentric video with overlaid 21-point hand skeletons | $1920 \times 1080$ preview |
| `output/hand_keypoints.json` | Per-frame structured JSON array of hand landmarks $(x, y, z)$ | 21 landmarks / hand |
| `output/camera_trajectory.csv` | Monocular camera $(X, Y, Z)$ poses per frame | Structured trajectory CSV |

---

## 5. Technical Pipeline Walkthrough

### Task 1 & 4: Video Inspection & Technical Properties
Using `cv2.VideoCapture`, all 8 MP4 containers were inspected:

| Stream | Filename | Resolution | FPS | Frame Count | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ego Head Left** | `HL_CH01_005_1013.MP4` | $1920 \times 1080$ | 29.97 | 2633 | ~87.85s |
| **Ego Head Right** | `HR_CH01_005_1013.MP4` | $1920 \times 1080$ | 29.97 | 2645 | ~88.25s |
| **Ego Wrist Left** | `WL_CH01_005_1013.MP4` | $1920 \times 1080$ | 29.97 | 2492 | ~83.15s |
| **Ego Wrist Right**| `WR_CH01_005_1013.MP4` | $1920 \times 1080$ | 29.97 | 2496 | ~83.28s |
| **UMI Gripper Left**| `GL_CH01_005_1014.MP4` | $1920 \times 1080$ | 29.97 | 2733 | ~91.19s |
| **UMI Gripper Right**| `GR_CH01_005_1014.MP4` | $1920 \times 1080$ | 29.97 | 2733 | ~91.19s |
| **UMI Head Left**   | `HL_CH01_005_1014.MP4` | $1920 \times 1080$ | 29.97 | 2720 | ~90.76s |
| **UMI Head Right**  | `HR_CH01_005_1014.MP4` | $1920 \times 1080$ | 29.97 | 2727 | ~90.99s |

*Key Takeaway*: All streams share a constant nominal frame rate of 29.97 FPS (NTSC standard) and identical resolution ($1920 \times 1080$). Durations vary slightly across devices by $\sim 1$–$5$ seconds due to asynchronous recording start/stop triggers.

### Task 2: Multi-Camera Synchronization
* **Alignment Approach**: Lock-step multi-stream reader. In each iteration, one frame is pulled from each camera capture object.
* **Layout**: Each $1920 \times 1080$ frame is downscaled to $640 \times 360$ and annotated with its camera tag using `cv2.putText`. Rows are concatenated using `np.hstack([HL, HR])` and `np.hstack([WL, WR])`, then stacked vertically via `np.vstack` into a 2x2 grid ($1280 \times 720$).
* **Limitations & Residual Uncertainty**: The raw MP4 containers lack SMPTE timecode tracks or hardware genlock signals. Frame alignment assumes simultaneous start; true inter-camera latency has an uncertainty of $\pm 1$–$2$ frames ($\sim 33$–$66$ ms).

### Task 3: Hand-Keypoint Extraction (MediaPipe + OpenCV)
* **Model**: MediaPipe Tasks Hand Landmarker bundle (`hand_landmarker.task`), which automatically downloads at runtime if not locally present.
* **Extraction**: 21 standard 3D coordinates $(x, y, z)$ normalized to $[0.0, 1.0]$.
* **Drawing**: Normalized coordinates are scaled to pixel space:
  $$\text{pixel\_x} = x \times W, \quad \text{pixel\_y} = y \times H$$
  Joints are drawn via `cv2.circle`, and anatomical connections (thumb, index, middle, ring, pinky, palm) are rendered with `cv2.line`.
* **Data Integrity**: Frames without detections are stored with an empty list (`"hands": []`) rather than omitted, preserving temporal index alignment.

### Task 5: Visual Odometry / Camera Trajectory Estimation
* **Intrinsic Camera Matrix ($K$)**: Because no factory calibration file was provided, a standard pinhole camera model was assumed:
  $$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} = \begin{bmatrix} 1000 & 0 & 960 \\ 0 & 1000 & 540 \\ 0 & 0 & 1 \end{bmatrix}$$
* **Feature Tracking**: Corner features detected with Shi-Tomasi (`cv2.goodFeaturesToTrack`) are tracked across successive frames using Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`).
* **Motion Estimation**: Essential Matrix $E$ is estimated using the 5-point algorithm with RANSAC outlier rejection (`cv2.findEssentialMat`), and camera rotation $R$ and translation $t$ are extracted via `cv2.recoverPose`.
* **Pose Accumulation**:
  $$R_{\text{cur}} = R_{\text{cur}} R, \quad t_{\text{cur}} = t_{\text{cur}} + R_{\text{cur}} t$$
* **Constraint Note**: Monocular visual odometry produces trajectories **up to an unknown scale factor**. Relative motion is physically consistent, but metric distances (meters) cannot be established without stereo baseline calibration or IMU integration.

---

## 6. Task 8: ML Readiness & Downstream Policy Training

### 1. Mapping into Downstream Policies (e.g., ACT, Diffusion Policy)
In imitation learning (e.g., LeRobot format):
* **Observations (`observation.*`)**:
  * `observation.images.head_left`: RGB frame ($3 \times H \times W$)
  * `observation.images.wrist_left`: Wrist perspective for fine manipulation
  * `observation.state`: Hand landmarks / end-effector 6-DoF pose ($x, y, z, \text{roll}, \text{pitch}, \text{yaw}$)
* **Actions (`action`)**:
  * Relative end-effector velocity $(\Delta x, \Delta y, \Delta z, \Delta \theta)$ or target gripper aperture $[0, 1]$.

### 2. IMU Integration & Visual-Inertial Odometry
The raw video containers do not contain embedded high-frequency IMU streams. In a full production robotics setup:
* A hardware IMU (accelerometer + gyroscope @ 200–500 Hz) provides metric scale and eliminates scale drift via extended Kalman filtering (e.g., VINS-Mono, ORB-SLAM3).
* Pre-integration of IMU samples between camera frames resolves rapid rotational blur where optical flow fails.

### 3. Data Leakage Prevention (Train / Val / Test Splits)
* **Rule**: Split data strictly by **episode / session / demonstrator**, **never randomly by frame**.
* Frames within the same recording share identical lighting, background, and visual features. Random frame-level splitting causes massive data leakage and misleadingly high validation metrics.

### 4. Measurable Validation Metrics
* **Pose Drift Rate**: Average relative translation error per meter traveled.
* **Hand Detection Consistency**: Proportion of continuous frames with $\ge 1$ hand detected and confidence $> 0.7$.
* **Imitation Policy Success**: Task completion rate (e.g., pick-and-place success) under closed-loop evaluation.

---

## 7. Assumptions, Known Limitations & Missing Inputs

1. **No Hardware Calibration Matrix**: Without checkerboard/ChArUco calibration, lens distortion (especially fisheye/barrel distortion on wide-angle head/wrist cams) cannot be rectified. Pinhole matrix $K$ is an approximation.
2. **Monocular Scale Ambiguity**: Poses in `camera_trajectory.csv` represent arbitrary scale units. Metric accuracy requires stereo baseline or metric sensor fusion.
3. **Absence of Action Labels & Gripper State**: Raw video alone lacks physical gripper force/aperture signals. For full imitation learning, synchronized robot state telemetry must be logged alongside video.
4. **Proprietary Data Compliance**: In accordance with Jupiter AI Labs confidentiality rules, all raw video recordings (`.MP4`) are excluded from version control via `.gitignore`.

---

## 8. AI Assistant Disclosure

AI coding assistant was used during this assessment for:
* Code structuring and OpenCV API synthesis.
* Research about the methods to approach the solution.
* Debugging dependency interactions between OpenCV, MediaPipe, and Protobuf on Python 3.12.
