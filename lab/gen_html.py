"""把教程 md 转成单文件 html（构建产物，不进版本库）。

用大白话说
    `knowledge/docs/` 里两份教程平时读 `.md` 就够了；要发给不做开发的人看，才需要 html。
    html 是**构建产物**：以前它跟着 md 一起进了工作区，两份内容重复约 74 KB，
    改一处忘了改另一处就成了两个版本。现在仓库只留 md，要 html 自己生成。

怎么跑（仓库根目录）
    python lab/gen_html.py                           # 转 knowledge/docs/ 下所有教程 md
    python lab/gen_html.py knowledge/docs/教程-从像素到毫米.md

放什么 / 不放什么
    只做"调用 pandoc 把 md 转成 html"这一件事；不生成目录、不套站点模板、
    不改 md 内容。排版细节交给 pandoc 默认样式（`--standalone`）。

依赖
    需要系统里有 `pandoc`（`sudo apt-get install pandoc`）。
    **故意不引入 markdown 库**：这条链路一阶段只跑一两次，不值得为它给运行时加依赖
    （与"不引 pydantic / 不引 click"同一个取舍）。

输出
    与 md 同名的 `.html`，落在同一个目录。`.gitignore` 已排除 `knowledge/docs/*.html`，
    生成后不会误提交；要对外发布就单独拷出去。
"""

import sys
from pathlib import Path

# `lab/` 不是包：直接 `python lab/xxx.py` 跑，脚本目录会进 sys.path，
# 但仓库根不会。这一行把仓库根补进去，才能 `import vision`（lab 可以依赖 vision，反过来不行）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import glob
import os
import shutil
import subprocess
import sys

#: 默认要转的教程（md）；找不到时按这个模式再扫一遍 knowledge/docs/
DEFAULT_PATTERN = os.path.join("knowledge", "docs", "教程-*.md")


def _pandoc_missing() -> int:
    """打印"缺 pandoc"的处理办法并返回退出码 2（环境错误，不是用法错误）。"""
    print("❌ 找不到 pandoc。它是 html 生成器，需要单独装：", file=sys.stderr)
    print("     Debian/Ubuntu: sudo apt-get install pandoc", file=sys.stderr)
    print("     其它平台: https://pandoc.org/installing.html", file=sys.stderr)
    print("   只想读文档不用装：knowledge/docs/ 里的 .md 直接用编辑器看即可。", file=sys.stderr)
    return 2


def convert(md_path: str) -> str:
    """把一个 md 转成同名 html。

    :param md_path: 输入的 md 路径
    :return: 写出的 html 路径
    :raises FileNotFoundError: md 不存在
    :raises subprocess.CalledProcessError: pandoc 转换失败（退出码非 0）
    """
    if not os.path.isfile(md_path):
        raise FileNotFoundError(f"找不到 {md_path}")
    html_path = os.path.splitext(md_path)[0] + ".html"
    # --standalone：产出可独立打开的完整 html（带 <head> 与内联样式），
    #   不加它得到的是一段 <body> 片段，双击打不开。
    # --metadata title：不给 title 时 pandoc 会警告并在页头留空。
    subprocess.run(
        ["pandoc", md_path, "-f", "gfm", "-t", "html5", "--standalone",
         "--metadata", f"title={os.path.splitext(os.path.basename(md_path))[0]}",
         "-o", html_path],
        check=True)
    return html_path


def main(argv=None) -> int:
    """按参数或默认模式转教程；打印每份产物的大小。

    :param argv: 参数列表；None 表示取 `sys.argv[1:]`
    :return: 0 全部成功 / 2 缺 pandoc 或没有可转的 md
    :raises subprocess.CalledProcessError: pandoc 转换失败时向上抛（由调用方看到真实退出码）
    """
    ap = argparse.ArgumentParser(prog="gen_html", description="教程 md -> 单文件 html")
    ap.add_argument("files", nargs="*", help=f"要转的 md；不给则用 {DEFAULT_PATTERN}")
    args = ap.parse_args(argv)

    if shutil.which("pandoc") is None:
        return _pandoc_missing()

    files = args.files or sorted(glob.glob(DEFAULT_PATTERN))
    if not files:
        print(f"❌ 没找到可转的 md（模式 {DEFAULT_PATTERN}）", file=sys.stderr)
        return 2

    for md_path in files:
        html_path = convert(md_path)
        size = os.path.getsize(html_path) / 1024.0
        print(f"✅ {md_path} -> {html_path}（{size:.1f} KB）")
    print("   提醒：html 是构建产物，已被 .gitignore 排除，不要提交")
    return 0


if __name__ == "__main__":
    sys.exit(main())
