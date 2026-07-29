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

**Built and why.** The repository vendors the pinned OpenArm v2 MuJoCo model,
meshes, license, and provenance under `vendor/openarm-v2`; setup does not fetch a
mutable external checkout. The scene retains the floor, table, rail collisions,
actuated vertical lifter, joint limits, and collision geometry. The roof and
front/side enclosure walls were removed to make the arms observable. The
lifter is commanded continuously to its legal upper limit, 0.3 m, but is not an
IK variable.

The 14 arm joints use the model's bounded position actuators and internal PD
gains. The controller writes desired positions to `data.ctrl` and advances
physics with `mujoco.mj_step`; it never animates the robot by overwriting
`qpos`. Physics uses the model's 1 ms step, while EGL rendering uses the WSL
D3D12/NVIDIA path rather than CPU `llvmpipe`.

**Considered instead.** OpenArm v1 exposed torque motors, which would require
designing and tuning an external PD/gravity-compensation controller. Converting
another URDF/MJCF would add model- and frame-validation risk. Fixing the
lifter's base was also considered, but it discarded an intentional actuator and
changed the upstream mechanism.

**Improve.** Validate inertial and friction parameters against a physical
OpenArm, add explicit self/environment collision limits to IK, and put physics
and rendering in separate processes so a slow render can never delay stepping.

## 2. Perception and monocular geometry

**Built and why.** A periodic YOLOX-Nano detector supplies a person ROI; the ROI
is reused between detections and refreshed early when confidence falls or the
body approaches its boundary. RTMW3D-L then estimates 133 whole-body landmarks
at 384 x 288 using a static-shape TensorRT FP32 engine. On the GTX 1060, the
pose-only benchmark is 33.1 ms mean and 38.6 ms p95; the full browser loop is
slower because it also contains JPEG, detector, drawing, and network work. The
highest-mean-confidence body instance is selected. Required landmarks must stay
above 0.35 for two continuous seconds before commands engage; any invalid frame
or 0.5 s camera timeout immediately holds the last target and restarts
acquisition.

When camera intrinsics are disabled, decoded SimCC values are converted into a
consistent proxy space: X and Y share the half-input-height scale and Z uses
RTMW3D's decoded relative-depth scale. Only limb directions are consumed.
When intrinsics are enabled, original-image points are undistorted and
backprojected. RTMW3D supplies root-relative depth; absolute root depth is
estimated from the two shoulder rays assuming 0.38 m shoulder width, bounded to
0.5-6 m, with a 2.5 m fallback. Camera profiles come from an automatic 12-view
ChArUco workflow and are tracked in Git; a centered 60-degree-HFOV,
zero-distortion model is the fallback.

**Considered instead.** RTMW3D-X with ONNX Runtime was more expensive with
negligible useful accuracy gain in this workload. MediaPipe is lighter but
offers weaker monocular depth and whole-body/hand coverage. A depth camera or
multi-view system would remove much of the scale ambiguity, but violates the
single-consumer-RGB-camera constraint and increases setup cost.

**Improve.** Track person identity instead of reselecting by confidence, reject
high-confidence temporal outliers, estimate body scale from several bones over
a window rather than one shoulder pair, and train/distill a task-specific
upper-body model. Calibrated stereo or RGB-D would be the highest-value sensing
upgrade.

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

After IK, a second typed boundary, `JointRetargetingTarget`, contains exactly
seven desired joint positions per arm in radians. This is the assignment-facing
desired robot state consumed by joint control; joint coordinates are intrinsic
to the OpenArm model, while the upstream `RobotTarget` preserves the required
base-frame Cartesian intent. The joint target is logged every control tick to
`assignment_logs/retargeting_target.jsonl`, including holds, source sequence,
mode, and tracking state. Perception targets and achieved state/solver
diagnostics are separately logged under `logs/`.

**Considered instead.** Neutral-pose-relative mapping was safer around the table
but failed the requirement for absolute imitation. Direct human-angle to
OpenArm-joint mapping could preserve redundancy cheaply, but human anatomical
axes do not coincide exactly with the robot's serial revolute axes and some
twist is unobservable without reliable hands. Unscaled Cartesian copying would
make commands depend on body size and camera distance.

**Improve.** Learn or calibrate a human-to-robot joint-space seed, then use it as
the posture reference for a small Cartesian correction. Add explicit reachable
workspace projection and confidence/covariance fields to the target contract.

## 4. IK and redundancy resolution

**Built and why.** Mink formulates one differential QP for both arms. Depending
on mode, it adds position tasks for the two elbow frames, two end-effector
sites, or both; orientation has lower weight and is opt-in. The QP is projected
onto exactly the 14 arm DoFs. The lifter and grippers participate in forward
kinematics but are absent from the decision vector and constraints.

Hard arm position and velocity limits bound the solve. A posture task anchored
to the previous solution resolves the remaining null space and discourages
branch changes; every solve warm-starts from that solution. The current
configuration uses a 0.7 task gain, 0.05 posture cost, 3 rad/s velocity limit,
and up to 25 QP steps. A bounded bimanual 3 cm test converges to approximately
3.8 mm residual in about 8-9 ms.

**Considered instead.** The earlier custom damped-least-squares solver was
simple and fast, but clipping after each update did not treat limits as part of
the optimization. Logs showed large joint changes near limits even for modest
Cartesian changes. Bounded nonlinear least squares would handle limits but has
less predictable real-time cost; analytical IK would be fastest but requires a
careful OpenArm-specific derivation and branch policy.

**Improve.** Add collision inequalities, manipulability and soft joint-centre
costs, per-task confidence weights, and acceptance logic that rejects a large
joint jump unless it materially reduces Cartesian error. A calibrated
joint-space seed would further stabilize redundant configurations.

## 5. Filtering and hold behavior

**Built and why.** A time-based exponential filter,
`alpha = 1 - exp(-dt/tau)`, smooths normalized upper-arm/forearm directions
before metric robot targets are constructed; the default `tau` is 0.25 s.
Rotation uses interpolation on SO(3). A second joint-space exponential stage is
available after IK but defaults to disabled (`tau = 0`); a 3 rad/s command-rate
limit remains active. This separates perception noise suppression from actuator
trajectory shaping. On confidence loss, the last actuator target is held rather
than extrapolated from uncertain measurements.

**Considered instead.** A fixed-alpha filter changes behavior with inference
rate. A Kalman or One Euro filter could reduce lag, but needs a trustworthy
measurement/noise model.

**Improve.** Add outlier rejection before the low-pass filter, velocity-aware
One Euro filtering, and timestamped prediction to command time. Uncertainty
should tune smoothing continuously instead of reducing confidence to a binary
engage/hold decision.

## 6. Threading, rate matching, and fast motion

**Built and why.** The application uses three spawned OS processes:
browser/FastAPI I/O, TensorRT perception/retargeting, and
MuJoCo/IK/control/rendering. Capacity-one latest-value queues discard stale
frames and targets, avoiding latency growth and the Python GIL. The configured
rates are 60 Hz control, 30 Hz render, and up to 24 browser captures per second;
MuJoCo performs as many 1 kHz steps as needed to follow wall time. The browser
currently permits only one camera frame in flight until the annotated response
returns, so debug rendering/network latency still caps perception intake.
Logged capture, inference, target, and simulation timestamps make this visible
rather than hiding it behind FPS.

**Considered instead.** A single threaded/`asyncio` process is simpler but lets
inference, rendering, and physics block one another. Unbounded queues increase
apparent throughput while controlling from stale poses. ROS or network services
would improve distributed observability but add serialization, deployment, and
failure modes without removing the unavoidable browser/server boundary.

**Improve and fast-motion limit.** First decouple camera submission from the
annotated-JPEG round trip and split render from the simulation owner. Add
deadline and queue-age telemetry. As motion gets faster, the first failure is
normally RTMW3D depth/ROI error or a high-confidence hypothesis jump; stronger
smoothing then adds visible lag. Next, large target steps encounter workspace,
joint-limit, or singularity constraints. Only after those failures do actuator
bandwidth and 1 kHz physics become limiting. The correct next version therefore
improves measurement continuity and latency before increasing controller
aggressiveness.
