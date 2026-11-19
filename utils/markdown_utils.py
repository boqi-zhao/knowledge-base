import re
from collections.abc import Iterator


def _closing_delimiter(text: str, start: int, opening: str, closing: str) -> int:
    depth = 1
    index = start + 1
    while index < len(text) and text[index] != "\n":
        char = text[index]
        if char == "\\" and text[index + 1:index + 2] not in ("", "\n"):
            index += 2
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def inline_image_spans(body: str) -> Iterator[tuple[int, int]]:
    """识别单行内联图片引用，保留转义字符和 URL 内成对的括号。"""
    end = 0
    for marker in re.finditer(r"\\[^\n]|!\[", body):
        if marker[0] != "![" or marker.start() < end:
            continue
        label_end = _closing_delimiter(body, marker.start() + 1, "[", "]")
        if label_end == -1 or not body.startswith("(", label_end + 1):
            continue
        destination_end = _closing_delimiter(body, label_end + 1, "(", ")")
        if destination_end != -1:
            end = destination_end + 1
            yield marker.start(), end


def strip_inline_images(text: str) -> str:
    """移除内联图片及替代文本，保留图片前后的文本边界。"""
    parts = []
    previous_end = 0
    for start, end in inline_image_spans(text):
        parts.append(text[previous_end:start])
        previous_end = end
    parts.append(text[previous_end:])
    return "\n".join(parts)
