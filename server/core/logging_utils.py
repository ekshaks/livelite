def clip_text(text, max_chars=1600):
    text = "" if text is None else str(text)
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n... [clipped {omitted} chars]"


def log_text_block(title, text, max_chars=1600):
    line = "=" * 16
    print(f"\n{line} {title} {line}")
    print(clip_text(text, max_chars=max_chars))
    print(f"{line} END {title} {line}\n")


def monitor_log(message):
    print(f"[monitor] {message}")
