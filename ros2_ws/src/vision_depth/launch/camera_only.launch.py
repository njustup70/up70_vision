"""camera_only.launch.py：只起相机，不带任何下游节点。

用大白话说
    把 336L 驱动起来，发深度图和彩色图。调试链路时单独用它，
    跑完整链路用 `fuse.launch.py`。

为什么点云默认关闭
    ROS2 接入方案第五节算过：1280×800 的 PointCloud2 是 16.4 MB/帧，
    30fps 就是 492 MB/s，会把 DDS 缓冲打满，表现是「话题有频率但数据不完整」，
    然后你会怀疑相机、怀疑驱动。融合节点订阅深度图、自己反投影。
    官方 `gemini_330_series.launch.py` 的 `enable_point_cloud` 默认是 **true**，
    所以这里必须显式关掉——照抄默认就会踩这个坑。

已知的坑
    · **`enable_frame_sync` 默认关。** 官方那个值默认是 true，同步器会等彩色帧；
      一旦关掉彩色，深度会**整条停发**——话题还在、就是不吐数据，很容易被误判
      成相机坏了或驱动挂了。所以纯深度调试时要 `enable_color:=false
      enable_frame_sync:=false` 一起给。同步本身也会增加延迟，第零段用不上。
    · `enable_depth_scale` 保持 true：关掉的话深度值单位会变，后面所有距离
      错一个量级，而且不报错（8 周计划也点名了这一条）。
    · 参数名随驱动版本变。换版本先 `ros2 param list` 核对，不要照抄教程。
    · 分辨率/帧率填 0 表示「用相机默认档」。336L 深度默认 640×400@30fps 左右，
      实测确认后写进 `config/camera_params.yaml`。
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    orbbec_share = get_package_share_directory("orbbec_camera")
    camera_launch = os.path.join(orbbec_share, "launch", "gemini_330_series.launch.py")

    args = [
        DeclareLaunchArgument("camera_name", default_value="camera",
                              description="话题前缀：/camera/... "),
        DeclareLaunchArgument("enable_color", default_value="true",
                              description="彩色流；关掉可以省带宽做纯深度调试"),
        DeclareLaunchArgument("enable_frame_sync", default_value="false",
                              description="帧同步：官方默认 true，关彩色时必须一起关，见文件头说明"),
        DeclareLaunchArgument("enable_left_ir", default_value="false",
                              description="左红外图流；A3 标深度内参要用它看棋盘格"),
        DeclareLaunchArgument("enable_laser", default_value="true",
                              description="激光投影；标定时必须关掉，否则棋盘上全是散斑"),
        DeclareLaunchArgument("depth_width", default_value="0",
                              description="深度宽，0 = 用相机默认档"),
        DeclareLaunchArgument("depth_height", default_value="0",
                              description="深度高，0 = 用相机默认档"),
        DeclareLaunchArgument("depth_fps", default_value="0",
                              description="深度帧率，0 = 用相机默认档"),
        DeclareLaunchArgument("color_width", default_value="0",
                              description="彩色宽，0 = 用相机默认档"),
        DeclareLaunchArgument("color_height", default_value="0",
                              description="彩色高，0 = 用相机默认档"),
        DeclareLaunchArgument("color_fps", default_value="0",
                              description="彩色帧率，0 = 用相机默认档"),
        DeclareLaunchArgument("enable_point_cloud", default_value="false",
                              description="点云：必须关，见文件头说明"),
        DeclareLaunchArgument("enable_colored_point_cloud", default_value="false",
                              description="彩色点云：同上"),
        DeclareLaunchArgument("depth_registration", default_value="false",
                              description="深度对齐到彩色；纯几何线不需要，开了要多算一遍"),
        DeclareLaunchArgument("enable_depth_scale", default_value="true",
                              description="深度值按 scale 换算成毫米；关掉会错一个量级"),
    ]

    # 只有出现在 launch_arguments 里的键才会被透传给被包含的 launch
    passthrough = [
        "camera_name", "depth_width", "depth_height", "depth_fps",
        "color_width", "color_height", "color_fps",
        "enable_color", "enable_frame_sync",
        "enable_left_ir", "enable_laser",
        "enable_point_cloud", "enable_colored_point_cloud",
        "depth_registration", "enable_depth_scale",
    ]

    return LaunchDescription(args + [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(camera_launch),
            launch_arguments={k: LaunchConfiguration(k) for k in passthrough}.items(),
        )
    ])
