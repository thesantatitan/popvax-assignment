# Vendored OpenArm v2 model

This directory contains the model and required mesh assets from
[`enactic/openarm_mujoco`](https://github.com/enactic/openarm_mujoco) commit
`8955afb54e4adfb59a236e2b4d15192b7a02865c`.

`cell.xml` was modified for this demo: the vertical lifter slide and its actuator
were removed, and the bimanual assembly was fixed at the midpoint of the original
`0–0.3 m` travel. The fixed body's world-height coordinate is therefore
`1.34 + 0.15 = 1.49 m`.

The upstream Apache License 2.0 is preserved in `LICENSE`.
