# Ultralytics YOLO26 Race training entrypoint.

from __future__ import annotations

import sys

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "detect", "train", *sys.argv[1:]]
    from ultralytics.cfg import entrypoint

    sys.exit(entrypoint())
