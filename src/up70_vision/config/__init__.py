"""配置层：配置结构定义、加载与校验。

放什么
    配置的 dataclass 定义、YAML 读取、命令行点分覆盖（--set key=value）的解析与校验。

不放什么
    不放业务默认值——默认值一律写在 configs/*.yaml 里，代码只定义结构与取值范围。

依赖谁
    只依赖 core.types（需要用到 Detection / Frame 等类型时）。

设计取舍
    用 dataclass + 显式校验函数，不引入 pydantic。理由是这个仓库要能被队友看懂——
    校验规则写在明面上，报错信息自己拼，比"读框架文档才知道为什么校验失败"更适合战队协作。

当前为空，未来第一件事
    写 config/schema.py：定义 AppConfig 及其子结构，每个字段带范围校验，
    非法取值时抛 ConfigError，报错信息里带上字段路径与可用值列表。
"""
