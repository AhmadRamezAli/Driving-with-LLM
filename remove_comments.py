import os
import re
import sys
from pathlib import Path

# Usage: python remove_comments.py [target_dir]
# Default target_dir is 'app/slices/prediction/ai'

EMOJI_PATTERN = re.compile(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]")


def contains_emoji(s: str) -> bool:
    return bool(EMOJI_PATTERN.search(s))


def process_file(path: Path):
    with path.open('r', encoding='utf-8') as f:
        lines = f.readlines()

    changed = False
    out_lines: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith('#'):
            # collect contiguous block of comment lines
            start = i
            block_has_emoji = contains_emoji(line)
            j = i
            while j + 1 < n and lines[j + 1].lstrip().startswith('#'):
                j += 1
                if contains_emoji(lines[j]):
                    block_has_emoji = True
            block_len = j - start + 1
            if block_has_emoji or block_len > 1:
                changed = True
                # skip this block
                i = j + 1
                continue
            else:
                # keep single-line comment with no emoji
                out_lines.append(line)
                i += 1
                continue
        else:
            out_lines.append(line)
            i += 1

    if changed:
        with path.open('w', encoding='utf-8') as f:
            f.writelines(out_lines)
        try:
            rel = path.relative_to(Path.cwd())
        except ValueError:
            rel = os.path.relpath(path, Path.cwd())
        print(f"Cleaned {rel}")


def main():
    target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('app/slices/prediction/ai')
    if not target_dir.exists():
        print(f"Directory {target_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    for py_file in target_dir.rglob('*.py'):
        process_file(py_file)


if __name__ == '__main__':
    main() 