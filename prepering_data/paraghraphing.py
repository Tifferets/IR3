import os
import re

INPUT_DIR = "UK_british_debates_text_files_normalize"
OUTPUT_DIR = "paragraph_chunks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# REGEX DEFINITIONS
# ===============================

SPEAKER_START = re.compile(
    r"^(Mr\.|Ms\.|Mrs\.|Miss\.|The Minister|The Secretary|The Prime Minister|Order\.|I call)\b"
)

PROCEDURAL_LINE = re.compile(
    r"^(Order\.|I call|Ordered,|Before we start|With permission|I thank|I welcome)\b"
)

def split_into_paragraphs(text: str):
    lines = [l.strip() for l in text.splitlines()]
    paragraphs = []
    buffer = []

    def flush():
        if buffer:
            paragraph = " ".join(buffer).strip()
            if len(paragraph) > 50:  # safety: ignore tiny junk
                paragraphs.append(paragraph)
            buffer.clear()

    for line in lines:
        if not line:
            flush()
            continue

        if PROCEDURAL_LINE.match(line):
            flush()
            paragraphs.append(line)
            continue

        if SPEAKER_START.match(line):
            flush()
            buffer.append(line)
            continue

        buffer.append(line)

    flush()
    return paragraphs


# ===============================
# MAIN LOOP
# ===============================

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".txt"):
        continue

    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, f"UK_{filename}")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    paragraphs = split_into_paragraphs(text)

    with open(output_path, "w", encoding="utf-8") as f:
        for p in paragraphs:
            f.write(p + "\n\n")

print("✅ Paragraph splitting complete — hierarchical chunk ready.")
