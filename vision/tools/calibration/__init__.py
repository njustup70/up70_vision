"""相机标定一族：打印棋盘、采图解内参。

放什么
    · `gen_board.py`  画棋盘 PNG + PDF（打印时选「实际大小」）
    · `calibrate.py`  现场采图 -> 找角点 -> 解内参与畸变 -> 写 YAML

不放什么
    不放标定算法（那是 `calibration.py`：`find_board_corners` / `calibrate` / `undistort_*`）。

怎么跑
    python -m vision gen-chessboard --cols 9 --rows 6 --square-mm 15
    python -m vision calibrate --capture 15 --board 9x6 --square-mm 23
"""
