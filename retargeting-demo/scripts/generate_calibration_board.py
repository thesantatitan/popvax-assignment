"""Generate the exact printable ChArUco target used by the web calibration tool."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retargeting_demo.camera_calibration import BOARD

DPI = 300
A4_PIXELS = (2480, 3508)
BOARD_PIXELS = (1984, 2835)


def main() -> None:
    board = BOARD.generateImage(BOARD_PIXELS, marginSize=0, borderBits=1)
    page = np.full((A4_PIXELS[1], A4_PIXELS[0]), 255, dtype=np.uint8)
    x = (A4_PIXELS[0] - BOARD_PIXELS[0]) // 2
    y = (A4_PIXELS[1] - BOARD_PIXELS[1]) // 2
    page[y : y + BOARD_PIXELS[1], x : x + BOARD_PIXELS[0]] = board
    image = Image.fromarray(page)
    image.save(ROOT / "calibration-board-a4-300dpi.png", dpi=(DPI, DPI))
    image.save(ROOT / "calibration-board-a4.pdf", resolution=DPI)


if __name__ == "__main__":
    main()
