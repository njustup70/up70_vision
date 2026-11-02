"""启动 aligned 相机与实时侧面查看器；回放时 start_camera:=false。

用大白话说
    相机档位沿用现有采集配置，但这里只启动相机和窗口，不启动录包器。
    两端保持仅共享内存传输；模型路径可从外层覆盖，不隐式下载或替换模型。
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, SetEnvironmentVariable, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _viewer(context):
    """显式 launch 参数覆盖 YAML；空串保留 YAML 值，避免默认值压过配置。"""
    overrides = {}
    weights = LaunchConfiguration('weights').perform(context)
    if weights:
        overrides['weights'] = weights
    headless = LaunchConfiguration('headless').perform(context)
    if headless:
        if headless.lower() not in ('true', 'false'):
            raise ValueError('headless must be true or false')
        overrides['headless'] = headless.lower() == 'true'
    duration = LaunchConfiguration('max_runtime_s').perform(context)
    if duration:
        overrides['max_runtime_s'] = float(duration)
    return [Node(package='vision_depth', executable='rgbd_side_viewer', output='screen',
                 parameters=[LaunchConfiguration('config_file').perform(context), overrides])]


def generate_launch_description():
    """注册查看器配置、模型路径及真机/回放开关。"""
    share = Path(get_package_share_directory('vision_depth'))
    config = LaunchConfiguration('config_file')
    return LaunchDescription([
        DeclareLaunchArgument('start_camera', default_value='true'),
        DeclareLaunchArgument('config_file', default_value=str(share/'config/rgbd_side_viewer.yaml')),
        DeclareLaunchArgument('camera_config', default_value=str(share/'config/rgbd_capture.yaml')),
        DeclareLaunchArgument('weights', default_value=''),
        DeclareLaunchArgument('headless', default_value=''),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('max_runtime_s', default_value=''),
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', str(share/'config/fastdds_shm_only.xml')),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('OPENBLAS_NUM_THREADS', '1'),
        # 子 launch 也叫 config_file；隔离作用域，避免相机配置覆盖查看器的配置。
        GroupAction(actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share/'launch/rgbd_capture.launch.py')),
            launch_arguments={'mode': 'aligned', 'config_file': LaunchConfiguration('camera_config')}.items())],
            scoped=True, condition=IfCondition(LaunchConfiguration('start_camera'))),
        OpaqueFunction(function=_viewer),
        Node(package='rviz2', executable='rviz2', name='rgbd_side_rviz',
             arguments=['-d', str(share/'config/rgbd_side.rviz')],
             condition=IfCondition(LaunchConfiguration('rviz')), output='screen'),
    ])
