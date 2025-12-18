import os
import re
import html
from pathlib import Path

INPUT_DIR = "US_congressional_speeches_Text_Files"
OUTPUT_DIR = "clean_txt"

Path(OUTPUT_DIR).mkdir(exist_ok=True)

DROP_EXACT = {
    "================================================================================",
    "____________________",
    "______",
    "<pre>",
    "</pre>",
    "[Extensions of Remarks]",
}

DROP_PREFIX = (
    "Title:",
    "Section:",
    "Date:",
    "Volume:",
    "Issue:",
    "Pages:",
)
DROP_PAGE_LINE = re.compile(r"^\[Page\s+[A-Z]\d+\]$")


DROP_CONTAINS = (
    "in the house of representatives",
)
DROP_ROOM_CODE = re.compile(r"^(SD|SR|SVC|SH)-[A-Z0-9]+$", re.IGNORECASE)

DROP_PAGE_BLOCK = re.compile(
    r"^\[+\s*Pages?\s+[A-Z]\d+(?:\s*-\s*[A-Z]?\d+)?\s*\]+$",
    re.IGNORECASE
)


DROP_STATE_LINE = re.compile(r"^of [a-z\s]+$", re.IGNORECASE)

DROP_DATE_LINE = re.compile(
    r"^(Monday|Tuesday|Wednesday|Thursday|Friday),\s+[A-Z][a-z]+\s+\d{1,2},\s+\d{4}"
)

def clean_text(text: str) -> str:
    text = html.unescape(text)
    cleaned = []

    for raw in text.splitlines():
        line = raw.strip()

        if not line:
            continue
        if line in DROP_EXACT:
            continue
        if line.startswith(DROP_PREFIX):
            continue
        if any(x in line.lower() for x in DROP_CONTAINS):
            continue
        if DROP_STATE_LINE.match(line):
            continue
        if DROP_DATE_LINE.match(line):
            continue
        if DROP_PAGE_LINE.match(line):
            continue
        if DROP_PAGE_BLOCK.match(line):
            continue
        if DROP_ROOM_CODE.match(line):
            continue

        cleaned.append(line)

    return "\n\n".join(cleaned)

def process_all():
    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith(".txt"):
            continue

        with open(Path(INPUT_DIR) / fname, encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        cleaned = clean_text(raw)

        with open(Path(OUTPUT_DIR) / fname, "w", encoding="utf-8") as out:
            out.write(cleaned)

        print(f"✅ Cleaned: {fname}")

if __name__ == "__main__":
    process_all()
