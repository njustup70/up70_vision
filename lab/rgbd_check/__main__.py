"""`python -m lab.rgbd_check ...` 的入口：转交 `cli.main()`。"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
