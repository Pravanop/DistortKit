#!/usr/bin/env python3
"""Turn the FAIL/ERROR blocks of a unittest log into GitHub annotations (visible on the run page without signing in).

    python tools/ci_annotate.py test.log
"""
import re
import sys


def esc(s):
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(path, limit=9, size=1800):
    log = open(path, errors="replace").read()
    blocks = re.split(r"\n={60,}\n", log)
    n = 0
    for b in blocks[1:]:
        b = b.strip()
        if not b.startswith(("FAIL:", "ERROR:")):
            continue
        lines = b.splitlines()
        head = lines[0]
        # header, then the traceback frames' last lines and the exception message (the test's own printed output can
        # follow the block and must not hide it)
        keep, seen_tb = [head], False
        for i, ln in enumerate(lines[1:], start=1):
            if ln.startswith("Traceback"):
                seen_tb = True
                keep.append(ln)
            elif seen_tb:
                keep.append(ln)
                if ln and not ln.startswith(" "):                 # the exception line
                    keep += [x for x in lines[i + 1:i + 6] if x.strip()][:4]
                    break
        text = "\n".join(keep)
        head_len = len(head)
        tail = text[head_len:][-(size - head_len):]
        print(f"::error title={esc(head[:120])}::{esc(head + chr(10) + tail)}")
        n += 1
        if n >= limit:
            break
    if not n:
        print("::notice::no FAIL/ERROR block found in the log; the run failed before or after the tests (see the log tail)")
        print(f"::error title=last lines of the log::{esc(chr(10).join(log.strip().splitlines()[-25:]))[:1800]}")


if __name__ == "__main__":
    main(sys.argv[1])
