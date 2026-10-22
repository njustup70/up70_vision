"""文档构建：把教程 md 转成单文件 html。

放什么
    `gen_html.py`：调用 pandoc 生成 `docs/教程-*.html`（构建产物，已被 .gitignore 排除）。

不放什么
    不套站点模板、不改 md 内容。

怎么跑
    python -m up70_vision.tools.doc.gen_html
"""
