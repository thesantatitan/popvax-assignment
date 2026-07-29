# Vendored OpenArm v2 model

This is the original model and required mesh set from
[`enactic/openarm_mujoco`](https://github.com/enactic/openarm_mujoco) commit
`8955afb54e4adfb59a236e2b4d15192b7a02865c`.

`cell.xml` removes the monolithic surrounding enclosure visual and the roof,
left/right-wall, and front-wall collision boxes. The separate tabletop visual,
table collision, lifter rails, floor, and robot remain. The robot definition itself
is unchanged: `openarm_lifter_joint` remains a `0–0.3 m` slide driven by the
`lifter_ctrl` position actuator, and the retargeting simulation continuously
commands that actuator to `0.3 m`.

The upstream Apache License 2.0 is preserved in `LICENSE`.
