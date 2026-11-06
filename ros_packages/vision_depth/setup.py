import os
from glob import glob

from setuptools import setup

package_name = "vision_depth"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        # Fast DDS 的仅 SHM 配置：大图走 UDP 分片会丢帧，本机采集两端都在同机，
        # 只走共享内存最稳。见 config/fastdds_shm_only.xml 里的说明。
        (os.path.join("share", package_name, "config"), glob("config/*.xml")),
        (os.path.join("share", package_name, "config"), glob("config/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="up70",
    maintainer_email="up70@example.com",
    description="RGB-D 可见平面候选识别与发布",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "rgbd_side_viewer = vision_depth.rgbd_side_viewer:main",
        ],
    },
)
