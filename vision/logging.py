"""日志配置：进程内调一次，其余模块用 get_logger() 取 logger。

设计取舍
    不引第三方日志库。战队协作要的是"打开 DEBUG 就能看到每帧的候选框数与各颜色
    命中率"，标准库 logging 完全够用，而且队友不用先学一个框架。

约定
    本包所有 logger 都挂在 "vision" 这个父 logger 下，
    因此 setup_logging() 调一次即可控制全包输出。
"""

import logging
import sys

PARENT_LOGGER = "vision"

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"

_configured = False


def parse_level(text: str) -> int:
    """把配置里的字符串等级转成 logging 等级。

    非法值直接抛 ValueError，并把可用值列在报错里——比默默用默认值好排查。
    """
    key = str(text).strip().upper()
    if key not in _LEVELS:
        raise ValueError(f"非法日志等级 {text!r}；可用值：{sorted(_LEVELS)}")
    return _LEVELS[key]


def setup_logging(level: str = "INFO", fmt: str = DEFAULT_FORMAT) -> logging.Logger:
    """配置并返回父 logger。重复调用只生效一次（幂等）。"""
    global _configured
    logger = logging.getLogger(PARENT_LOGGER)
    logger.setLevel(parse_level(level))
    if not _configured:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.propagate = False
        _configured = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """取子 logger，例如 get_logger("color") -> vision.color"""
    suffix = name.strip(".")
    return logging.getLogger(f"{PARENT_LOGGER}.{suffix}" if suffix else PARENT_LOGGER)
