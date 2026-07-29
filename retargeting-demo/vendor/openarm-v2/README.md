# Vendored OpenArm v2 model

This is the original model and required mesh set from
[`enactic/openarm_mujoco`](https://github.com/enactic/openarm_mujoco) commit
`8955afb54e4adfb59a236e2b4d15192b7a02865c`.

The MuJoCo definitions are unmodified. In particular, `openarm_lifter_joint`
remains a `0–0.3 m` slide driven by the `lifter_ctrl` position actuator. The
retargeting simulation continuously commands that actuator to `0.3 m`.

The upstream Apache License 2.0 is preserved in `LICENSE`.
