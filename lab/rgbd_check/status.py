"""四项验收状态的统一写法与合并规则。

用大白话说
    报告里每一项只有四种结论：过了、没过、没验证、不适用。代码里分散写字符串
    最容易出现 `pass` / `PASS` / `通过` 三种写法，下游一读就错，所以集中在这里。

为什么要区分 UNVERIFIED 和 FAIL
    没数据 ≠ 数据不合格。缺标注、没采这段、相机没接，都是「没验证」；
    测了但超门槛才是「没过」。把前者写成 FAIL 会让人去查一个不存在的故障，
    把后者写成 UNVERIFIED 则会拿没测当合格。
"""

PASS = "PASS"
FAIL = "FAIL"
UNVERIFIED = "UNVERIFIED"
SKIP = "SKIP"

#: 从坏到好的顺序，`worst()` 按这个排。
_SEVERITY = {FAIL: 3, UNVERIFIED: 2, SKIP: 1, PASS: 0}


def worst(statuses) -> str:
    """取一组状态里最差的那个；空集合视为 UNVERIFIED（什么都没测）。"""
    items = [s for s in statuses if s]
    if not items:
        return UNVERIFIED
    return max(items, key=lambda s: _SEVERITY.get(s, 2))
