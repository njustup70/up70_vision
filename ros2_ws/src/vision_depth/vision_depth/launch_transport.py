"""为相机与查看器共用的启动入口选择通信环境。

默认保留 Fast DDS 仅 SHM；集成时选 inherit，不改主容器的 RMW、路由或 domain。
"""
from launch.actions import SetEnvironmentVariable

TRANSPORT_CHOICES = ('fastdds_shm', 'inherit')


def transport_actions(transport, profile):
    """返回需要修改的 launch 环境；inherit 返回空列表。

    :param transport: fastdds_shm 或 inherit，拼错时拒绝启动
    :param profile: Fast DDS XML 的绝对路径，仅默认模式使用
    :return: 环境设置 action 列表；不直接修改当前进程环境
    """
    if transport not in TRANSPORT_CHOICES:
        raise ValueError(f'transport={transport!r}; expected {TRANSPORT_CHOICES}')
    if transport == 'inherit':
        return []
    return [SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
            SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', str(profile))]
