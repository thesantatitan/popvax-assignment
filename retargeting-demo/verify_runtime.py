"""Verify the combined demo's model, GPU providers, and TensorRT engine."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mujoco
import onnxruntime as ort
from runtime_paths import tensorrt_engine_path

ROOT = Path(__file__).resolve().parent
MODEL = Path(
    os.getenv(
        "OPENARM_MODEL_PATH",
        str(
            ROOT
            / "vendor"
            / "openarm-v2"
            / "cell.xml"
        ),
    )
)
ENGINE = tensorrt_engine_path()


def main() -> int:
    problems: list[str] = []
    if not MODEL.is_file():
        problems.append(f"OpenArm model is missing: {MODEL}")
    else:
        model = mujoco.MjModel.from_xml_path(str(MODEL))
        if model.nu < 14:
            problems.append(f"Expected at least 14 actuators, found {model.nu}")
        lifter_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lifter_ctrl"
        )
        if lifter_id < 0:
            problems.append("OpenArm lifter_ctrl actuator is missing")
        elif model.actuator_ctrlrange[lifter_id, 1] != 0.3:
            problems.append(
                "Expected lifter upper control limit 0.3 m, found "
                f"{model.actuator_ctrlrange[lifter_id, 1]}"
            )
    if sys.platform.startswith("linux"):
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            problems.append(f"ONNX Runtime CUDA provider is missing: {providers}")
        if not ENGINE.is_file():
            problems.append(f"TensorRT engine is missing: {ENGINE}")
    if problems:
        print("Runtime verification failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print(f"OpenArm model: {MODEL}")
    print(f"ONNX Runtime {ort.__version__}: {ort.get_available_providers()}")
    if sys.platform.startswith("linux"):
        print(f"TensorRT engine: {ENGINE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
