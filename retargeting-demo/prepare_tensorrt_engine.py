"""Download or locally build the RTMW3D-L TensorRT 8.6 FP32 engine."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

from runtime_paths import tensorrt_engine_path

CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/"
    "rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth"
)
MMPOSE_REPOSITORY = "https://github.com/open-mmlab/mmpose.git"
MMPOSE_TAG = "v1.3.2"


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}")
    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _stub_optional_imports() -> None:
    xt = types.ModuleType("xtcocotools")
    xt.__path__ = []  # type: ignore[attr-defined]
    coco = types.ModuleType("xtcocotools.coco")
    coco.COCO = type("COCO", (), {})
    cocoeval = types.ModuleType("xtcocotools.cocoeval")
    cocoeval.COCOeval = type("COCOeval", (), {})
    mask = types.ModuleType("xtcocotools.mask")
    ops = types.ModuleType("mmcv.ops")
    ops.MultiScaleDeformableAttention = type(
        "MultiScaleDeformableAttention", (), {}
    )
    sys.modules.update(
        {
            "xtcocotools": xt,
            "xtcocotools.coco": coco,
            "xtcocotools.cocoeval": cocoeval,
            "xtcocotools.mask": mask,
            "mmcv.ops": ops,
        }
    )


def _export_onnx(work_dir: Path, onnx_path: Path) -> None:
    import torch

    source = work_dir / "mmpose"
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--depth",
            "1",
            "--branch",
            MMPOSE_TAG,
            MMPOSE_REPOSITORY,
            str(source),
        ],
        check=True,
    )
    checkpoint = work_dir / Path(CHECKPOINT_URL).name
    _download(CHECKPOINT_URL, checkpoint)

    sys.path[:0] = [str(source / "projects" / "rtmpose3d"), str(source)]
    _stub_optional_imports()
    from mmengine.config import Config
    from mmengine.model.utils import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    import mmpose.models  # noqa: F401
    import rtmpose3d  # noqa: F401
    from mmpose.models.builder import build_pose_estimator

    config = Config.fromfile(
        str(
            source
            / "projects"
            / "rtmpose3d"
            / "configs"
            / "rtmw3d-l_8xb64_cocktail14-384x288.py"
        )
    )
    config.model.train_cfg = None
    init_default_scope("mmpose")
    model = revert_sync_batchnorm(build_pose_estimator(config.model))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"])
    model.eval()

    class ExportWrapper(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, value: torch.Tensor) -> object:
            return self.wrapped(value, None, mode="tensor")

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            ExportWrapper(model).eval(),
            torch.randn(1, 3, 384, 288),
            onnx_path,
            input_names=["input"],
            output_names=["simcc_x", "simcc_y", "simcc_z"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )


def _build_engine(onnx_path: Path, engine_path: Path) -> None:
    import tensorrt_bindings as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse the exported ONNX model:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    print("Building the FP32 TensorRT engine; this can take several minutes.")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the RTMW3D-L engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = engine_path.with_suffix(".plan.part")
    temporary.write_bytes(serialized)
    temporary.replace(engine_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rebuild an existing engine")
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        print("TensorRT engine preparation is only supported on Linux/WSL.")
        return 0

    engine_path = tensorrt_engine_path()
    if engine_path.is_file() and not args.force:
        print(f"TensorRT engine already exists: {engine_path}")
        return 0

    engine_url = os.getenv("RTMW3D_TRT_ENGINE_URL")
    if engine_url:
        _download(engine_url, engine_path)
        print(f"TensorRT engine downloaded to: {engine_path}")
        return 0

    configured_onnx = os.getenv("RTMW3D_ONNX_MODEL")
    with tempfile.TemporaryDirectory(prefix="rtmw3d-engine-") as temporary:
        work_dir = Path(temporary)
        onnx_path = (
            Path(configured_onnx).expanduser()
            if configured_onnx
            else work_dir / "rtmw3d-l-384x288.onnx"
        )
        if not onnx_path.is_file():
            _export_onnx(work_dir, onnx_path)
        _build_engine(onnx_path, engine_path)
    print(f"TensorRT engine ready: {engine_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
