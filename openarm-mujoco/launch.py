"""Launch the pinned OpenArm v1 bimanual scene in MuJoCo Viewer."""

from pathlib import Path

import mujoco
import mujoco.viewer


MODEL_PATH = (
    Path(__file__).resolve().parent
    / ".assets"
    / "openarm_mujoco"
    / "v1"
    / "openarm_bimanual.xml"
)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()

