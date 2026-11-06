"""相机与查看器通信选择回归，不启动任何节点或设备。"""
import importlib.util
import os
from pathlib import Path
import sys

import pytest

pytest.importorskip('launch')
from launch import LaunchContext, LaunchService
from launch.actions import OpaqueFunction, SetLaunchConfiguration
from launch.substitutions import LaunchConfiguration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ros2_ws/src/vision_depth'))
from vision_depth.launch_transport import transport_actions


def load_launch(name):
    """读取实际 launch，使用假 driver action 避免启动设备。"""
    spec = importlib.util.spec_from_file_location(name, ROOT/f'ros2_ws/src/vision_depth/launch/{name}.launch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('rmw', ['rmw_zenoh_cpp', 'rmw_cyclonedds_cpp'])
def test_inherit_does_not_modify_parent_environment(rmw):
    """继承模式完全保留 domain、RMW 和已有 profile。"""
    context = LaunchContext()
    context.environment.update(RMW_IMPLEMENTATION=rmw, ROS_DOMAIN_ID='73',
                               FASTRTPS_DEFAULT_PROFILES_FILE='/parent/profile.xml')
    before = dict(context.environment)
    for action in transport_actions('inherit', '/new/profile.xml'):
        action.execute(context)
    assert dict(context.environment) == before


def test_default_sets_fastdds_only_and_rejects_unknown_transport():
    """默认模式明确设 Fast DDS/SHM，但不改 domain；拼错通信模式必须报错。"""
    context = LaunchContext()
    context.environment.update(RMW_IMPLEMENTATION='rmw_zenoh_cpp', ROS_DOMAIN_ID='73')
    for action in transport_actions('fastdds_shm', '/new/profile.xml'):
        action.execute(context)
    assert context.environment['RMW_IMPLEMENTATION'] == 'rmw_fastrtps_cpp'
    assert context.environment['FASTRTPS_DEFAULT_PROFILES_FILE'] == '/new/profile.xml'
    assert context.environment['ROS_DOMAIN_ID'] == '73'
    with pytest.raises(ValueError, match='transport'):
        transport_actions('typo', '/new/profile.xml')


@pytest.mark.parametrize('transport', ['inherit', 'fastdds_shm'])
def test_capture_resolver_respects_transport(monkeypatch, transport):
    """真实相机 resolver 继承或设置环境，不被内部 SHM 设置再次覆盖。"""
    module = load_launch('rgbd_capture')
    share = ROOT/'ros2_ws/src/vision_depth'
    monkeypatch.setattr(module, 'get_package_share_directory', lambda name: str(share))
    monkeypatch.setattr(module, 'PythonLaunchDescriptionSource', lambda *args: None)
    monkeypatch.setattr(module, 'IncludeLaunchDescription', lambda *a, **k: OpaqueFunction(function=lambda c: []))
    context = LaunchContext()
    context.environment['RMW_IMPLEMENTATION'] = 'rmw_zenoh_cpp'
    context.environment['FASTRTPS_DEFAULT_PROFILES_FILE'] = '/parent.xml'
    context.launch_configurations.update({key: '' for key in module.PASSTHROUGH_ARGS})
    context.launch_configurations.update(transport=transport, mode='aligned', config_file=str(share/'config/rgbd_capture.yaml'))
    for action in module._resolve(context):
        action.execute(context)
    expected = 'rmw_zenoh_cpp' if transport == 'inherit' else 'rmw_fastrtps_cpp'
    assert context.environment['RMW_IMPLEMENTATION'] == expected
    assert context.environment['FASTRTPS_DEFAULT_PROFILES_FILE'] == (
        '/parent.xml' if transport == 'inherit' else str(share/'config/fastdds_shm_only.xml'))


@pytest.mark.parametrize('transport', ['inherit', 'fastdds_shm'])
def test_viewer_forwards_transport_with_scoped_camera_config(monkeypatch, transport):
    """相机子入口收到同一个 transport，子配置不会覆盖查看器配置。"""
    module = load_launch('rgbd_side_viewer')
    share = '/tmp/handoff-share'
    seen = []
    monkeypatch.setattr(module, 'get_package_share_directory', lambda name: share)
    monkeypatch.setattr(module, 'PythonLaunchDescriptionSource', lambda *args: None)

    def include(*args, launch_arguments):
        """模拟子 launch，但记录实参并改动自己的 config_file。"""
        arguments = dict(launch_arguments)

        def child(context):
            """在真实 launch 上下文中求 transport。"""
            seen.append(LaunchConfiguration('transport').perform(context))
            assert arguments['transport'].perform(context) == transport
            return [SetLaunchConfiguration('config_file', '/child.yaml')]
        return OpaqueFunction(function=child)

    monkeypatch.setattr(module, 'IncludeLaunchDescription', include)
    monkeypatch.setattr(module, '_viewer', lambda context: seen.append((
        LaunchConfiguration('config_file').perform(context), context.environment['RMW_IMPLEMENTATION'])) or [])
    service = LaunchService()
    description = module.generate_launch_description()
    description.entities.insert(0, SetLaunchConfiguration('transport', transport))
    monkeypatch.setenv('RMW_IMPLEMENTATION', 'rmw_zenoh_cpp')
    service.include_launch_description(description)
    assert service.run() == 0
    assert seen == [transport, (share+'/config/rgbd_side_viewer.yaml',
                               'rmw_zenoh_cpp' if transport == 'inherit' else 'rmw_fastrtps_cpp')]
