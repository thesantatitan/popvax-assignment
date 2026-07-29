# Webcam-to-OpenArm Retargeting: Design Document

## System overview and development platform

The system turns monocular RGB video from the browser into actuator commands for
a bimanual OpenArm v2 model:

```text
browser camera -> RTMW3D pose -> robot-base Cartesian target
               -> Mink IK -> 14-joint target -> data.ctrl -> MuJoCo
```

The browser captures the camera on the device displaying the page, so the
compute server may run in WSL while the camera remains on the Windows host or a
Mac connected through an SSH tunnel. Development used Ubuntu under WSL, a GTX
1060 6 GB, Python 3.11 managed by `uv`, TensorRT 8.6, and MuJoCo 3.6. The web UI
shows the annotated camera image, a 3D pose inset, the MuJoCo render, tracking
state, Cartesian errors, and filtered end-effector traces.

## 1. Scene construction and actuation

The repository vendors the pinned OpenArm v2 MuJoCo model,
meshes, license, and provenance under `vendor/openarm-v2`; setup does not fetch a
mutable external checkout. The scene had an enclosure and a vertical linear joint that could move the arms upto 0.3m. I removed the enclosure walls and roof, as they were interfering with full arm extension. Also the lifter is constantly commanded to 0.3m position because other wise the arms were hitting the table

The 14 arm joints use the model's bounded position actuators and internal PD
gains. The controller writes desired positions to `data.ctrl` and advances
physics with `mujoco.mj_step`. The internal PD controller moves the actuators to desried positions. Physics uses the model's 1 ms step, while EGL rendering uses the WSL
D3D12/NVIDIA path. 



## 2. Perception and monocular geometry

**Built and why.** A YOLOX-Nano detector is first used for calculating person ROI whcih is then rescaled to 384x288 to pass to pose estimator. The ROI detector runs every 10 frames or if the pose starts getting bad.
RTMW3D-L then estimates 133 whole-body keypoints using a static TensorRT fp32 engine. On my GTX 1060 this give a reliable 15fps pose estimation end to end.

RTMW3D returns a weird coordinate space called SimCC. X and Y coordinate are in cropped image space, and Z is relative to root joint(mean of hips) in a range of +-2.17 units. 

For recovering the correct 3d pose from these units camera intrinsics would be helpful but a close enough approximation can be achieved without them

### Without camera intrinsics

Decoded SimCC values are converted into a consistent proxy space. z is multiplied by 2.17 because during training it was normalized to this range

$$
x_c = \frac{x_s}{192}\\
$$
$$
y_c = \frac{y_s}{192}\\
$$
$$
z_c = \frac{z_s}{192} \times 2.1744869
$$

### With camera intrinsics

When Camera intrinsics are enabled in the UI, original image points are undestorted and back projected into metric 3d space. The root depth is estimated by assuming shoulder width to be 0.38m and then each keypoint is converted to metric camera space.

Once we have the coordinates in camera space, elbow and end effector directions are calculated and then scaled by openarm given lengths to get desired positions

**Considered instead.** RTMW3D-X with ONNX Runtime was more expensive with
negligible useful accuracy gain in this workload. MediaPipe is lighter but
offers weaker monocular depth and whole-body/hand coverage. A depth camera or
multi-view system would remove much of the scale ambiguity, but violates the
single-consumer-RGB-camera constraint and increases setup cost. Also considered FastSam3dBody. This is especially interesting since they have a demo of teleoping a humanoid by monocular camera. But this was agin too big to run in my GPU.

**Improve.** I wanna try, several things here, first is obviously bigger models, but also distillation of bigger models for the specific keypoints we are interested in to smaller models. So we can have both speed and accuracy.

## 3. Retargeting and the intermediate target

**Built and why.** Perception produces a typed `RobotTarget` containing left and
right elbow positions, optional wrist position/orientation, confidence,
timestamps, mode, and calibration metadata. Positions are in metres in the
OpenArm `arm_origin` frame. The camera-to-robot transform maps image right to
robot left, image down to robot down, and camera depth to the opposite of robot
forward.

For each arm, shoulder-to-elbow and elbow-to-wrist vectors are normalized.
Human limb lengths and image translation are discarded; the vectors are rebuilt
from fixed robot shoulders using OpenArm's 0.220 m upper arm and 0.216 m forearm.
This is an absolute pose mapping with no neutral-pose calibration, so it is
body-size invariant and does not reinterpret the operator's starting pose as
robot home. The UI can target elbow only, end effector only, both, or both plus
hand-derived orientation.

Right now hand-derived wrist orientation doesn't work very well, but it doesn't seem to be a big issue and can probably be fixed improvements in IK

## 4. IK and redundancy resolution

**Built and why.** Mink formulates one differential QP for both arms. Depending
on mode, it adds position tasks for the two elbow frames, two end-effector
sites, or both; orientation has lower weight and is opt-in. The QP is projected
onto exactly the 14 arm DoFs. The lifter and grippers participate in forward
kinematics but are absent from the decision vector and constraints.

**Considered instead.** A basic least squares IK solver was also tried but it failed aroung singularities. 

**Improve.** Since OpenArm closely mimics human joints, I want to try an analytical joint aware IK solver, instead of a generic one like right now. Will probably be faster and more deterministic with more human-like movements
