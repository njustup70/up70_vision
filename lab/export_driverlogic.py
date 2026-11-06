"""导出 driverLogic_2027 所需的 RGB-D 运行模块和只读接入文件。

用大白话说
    从当前工作副本挑选运行源码，不复制整个仓库。输出必须是新目录。
    模型可选单独导出到 model_assets；MANIFEST.json 与 SHA256SUMS 记录实际文件，
    不是只凭 Git 提交号假定工作副本干净。此工具不修改目标主逻辑仓库。
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DRIVERLOGIC_REVISION = 'e5b2a2c4fb31e9cce12b1452a0e8459f93732514'
MODEL_FILES = ('yolov8n_mix4.onnx', 'yolov8n_mix4.names.json')


def _copy(source, target):
    """复制一个普通文件；拒绝符号链接，避免把仓库外的数据带进交接包。"""
    if source.is_symlink() or not source.is_file():
        raise ValueError(f'expected a regular source file: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _tree(source, target, suffixes=None):
    """复制运行目录中的文件，排除缓存；suffixes 为 None 时保留普通文件。"""
    if source.is_symlink():
        raise ValueError(f'symlink source directory: {source}')
    for path in sorted(source.rglob('*')):
        relative = path.relative_to(source)
        if any(part.startswith('.') or part == '__pycache__' or part.endswith('.egg-info')
               for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f'symlink source: {path}')
        if path.is_file() and (suffixes is None or path.suffix in suffixes):
            _copy(path, target/relative)


def _git(*args):
    """读取版本或工作区状态；非 Git 源码目录也可以导出，但版本记为 unavailable。"""
    result = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else 'unavailable'


def export_bundle(out, model_dir=None):
    """生成全新交接目录，失败时删除本次创建的不完整目录。

    :param out: 不得已存在的输出目录；存在时抛 FileExistsError
    :param model_dir: 可选模型来源目录，两个配套文件分别导出到 model_assets
    :return: 输出目录的绝对路径；写入 manifest 和每个交付文件的 SHA-256
    """
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'output already exists: {out}')
    for source in (ROOT/'vision', ROOT/'ros2_ws/src', ROOT/'integration/driverlogic'):
        if out.resolve().is_relative_to(source.resolve()):
            raise ValueError(f'output must not be inside a copied source tree: {out}')
    if model_dir is not None:
        model_dir = Path(model_dir).resolve()
        for name in MODEL_FILES:
            source = model_dir/name
            if source.is_symlink() or not source.is_file() or not source.stat().st_size:
                raise ValueError(f'missing or invalid model asset: {source}')
        names = json.loads((model_dir/MODEL_FILES[1]).read_text(encoding='utf-8'))
        if names not in ({'0': 'block', '1': 'top'}, ['block', 'top']):
            raise ValueError('mix4 names must be 0=block, 1=top')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    try:
        runtime = out/'runtime'
        _tree(ROOT/'vision', runtime/'vision', {'.py'})
        _copy(ROOT/'pyproject.toml', runtime/'pyproject.toml')
        guide = ROOT/'integration/driverlogic/README.md'
        _copy(guide, out/'README.md')
        _copy(guide, runtime/'README.md')  # 原 pyproject 的 readme 元数据依赖。
        _copy(ROOT/'integration/driverlogic/requirements-humble.txt',
              runtime/'requirements-humble.txt')
        _copy(ROOT/'configs/color_thresholds.yaml', runtime/'configs/color_thresholds.yaml')
        _copy(ROOT/'configs/README.md', runtime/'configs/README.md')
        for package in ('vision_depth', 'vision_interfaces'):
            source = ROOT/'ros2_ws/src'/package
            target = runtime/'ros2_ws/src'/package
            for name in ('package.xml', 'setup.py', 'setup.cfg', 'CMakeLists.txt'):
                if (source/name).exists():
                    _copy(source/name, target/name)
            for name in ('vision_depth', 'launch', 'config', 'resource', 'msg'):
                if (source/name).is_dir():
                    _tree(source/name, target/name)
        _tree(ROOT/'integration/driverlogic/MainLogic', out/'mainlogic/MainLogic', {'.py'})
        _copy(ROOT/'vision/docs/interface.md', out/'docs/interface.md')
        _copy(ROOT/'vision/depth/README.md', out/'docs/depth-README.md')
        (runtime/'knowledge/models').mkdir(parents=True)
        (runtime/'COLCON_IGNORE').touch()
        (runtime/'.gitignore').write_text('*.onnx\n*.names.json\noutput/\n*.egg-info/\n__pycache__/\n',
                                         encoding='utf-8')
        assets = []
        if model_dir is not None:
            for name in MODEL_FILES:
                _copy(model_dir/name, out/'model_assets'/name)
                assets.append(name)
        status = _git('status', '--porcelain')
        manifest = {'format_version': 1, 'vision_revision': _git('rev-parse', 'HEAD'),
                    'source_worktree_dirty': None if status == 'unavailable' else bool(status),
                    'driverlogic_branch': 'driverLogic_2027',
                    'driverlogic_revision': DRIVERLOGIC_REVISION,
                    'ros_distro': 'humble', 'python': '3.10',
                    'transport': 'inherit parent rmw_zenoh_cpp', 'model_assets': assets, 'files': []}
        for path in sorted(out.rglob('*')):
            if path.is_file():
                manifest['files'].append({'path': path.relative_to(out).as_posix(),
                                          'size': path.stat().st_size,
                                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        (out/'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n',
                                         encoding='utf-8')
        paths = [*manifest['files'], {'path': 'MANIFEST.json',
                 'sha256': hashlib.sha256((out/'MANIFEST.json').read_bytes()).hexdigest()}]
        (out/'SHA256SUMS').write_text(''.join(f"{row['sha256']}  {row['path']}\n" for row in paths),
                                     encoding='utf-8')
    except Exception:
        # 仅删除已由本调用成功 mkdir 的目录；清理后保留原异常供 CLI 或调用者定位。
        shutil.rmtree(out)
        raise
    return out


def main(argv=None):
    """解析导出参数，成功返回 0、输入或文件错误返回 2。"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True, help='new output directory; never overwritten')
    parser.add_argument('--model-dir', help='optional directory with both mix4 model files')
    args = parser.parse_args(argv)
    try:
        out = export_bundle(args.out, args.model_dir)
    except (OSError, ValueError) as exc:
        print(f'Export failed: {exc}')
        return 2
    print(f'Exported RGBD handoff: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
