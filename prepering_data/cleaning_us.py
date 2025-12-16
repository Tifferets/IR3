
import os
import re
import html
import unicodedata

# ===============================
# CONFIG
# ===============================

INPUT_DIR = "US_congressional_speeches_Text_Files"
OUTPUT_DIR = "cleaned_congressional_text"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# REGEX DEFINITIONS (GLOBAL)
# ===============================

# ALL CAPS speech titles
TITLE_RE = re.compile(r"^[A-Z][A-Z0-9 ,.'\-()]{8,}$")

# Lines to REMOVE completely (robust against Unicode colons and tags)
REMOVE_LINE_PATTERNS = [
    r"^\s*Title\s*[:：].*$",
    r"^\s*Section\s*[:：].*$",
    r"^\s*Date\s*[:：].*$",
    r"^\s*Volume\s*[:：].*$",
    r"^\s*Issue\s*[:：].*$",
    r"^\s*Pages\s*[:：].*$",
    r"^\s*Congressional Record.*$",
    r"^\s*From the Congressional Record.*$",
    r"^\s*\[Extensions of Remarks\]\s*$",
    r"^\s*\[Page.*\]\s*$",
    r"^\s*\[\[Page.*\]\]\s*$",
    r"^\s*&lt;.*?&gt;\s*$",     # pre-decoding HTML entities (if any survive)
    r"^\s*<.*?>\s*$",          # post-decoding HTML tags on their own lines
    r"^\s*=+\s*$",             # pure separator lines
    r"^\s*_+\s*$",
    r"^\s*=+\s*[A-Z ]+\s*=+\s*$",  # e.g., "===== NOTE =====", "===== END NOTE ====="
    r"^\s*HON\..*$",
    r"^\s*of [a-zA-Z .'\-]+$",     # name tails like "of Texas" alone
    r"^\s*in the house of representatives\s*$",
    r"^\s*[A-Za-z]+,\s+[A-Za-z]+\s+\d{1,2},\s+\d{4}\s*$",  # e.g., Monday, January 12, 2020
]
REMOVE_RE = re.compile("|".join(REMOVE_LINE_PATTERNS), re.IGNORECASE)

# Speaker prefix list
SPEAKER_PREFIXES = (
    "Mr.", "Ms.", "Mrs.", "Miss.",
    "Madam Speaker", "The SPEAKER",
    "Representative", "Senator"
)


# ===============================
# CLEAN FUNCTION
# ===============================

def clean_file(text: str) -> str:
    """
    Clean congressional text:
    - Normalize Unicode and decode HTML entities
    - Remove metadata/noise lines
    - Split titles as hard boundaries
    - Reconstruct paragraphs
    - Drop NOTE sections entirely
    - Stop at 'SENATE COMMITTEE MEETINGS'
    - Handle mid-line speaker starts
    """
    # Normalize & decode
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    # Replace non-breaking spaces with normal spaces
    text = text.replace("\u00A0", " ")

    # De-hyphenate line breaks (word split across lines like "ex-\nample")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Drop NOTE sections completely (header to footer, multiline)
    text = re.sub(
        r"\n=+\s*NOTE\s*=+\n.*?\n=+\s*END NOTE\s*=+\n",
        "\n",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # HARD STOP before Senate schedules (case-insensitive)
    text = re.split(r"\nSENATE COMMITTEE MEETINGS\n", text, maxsplit=1, flags=re.IGNORECASE)[0]

    # Work line-by-line
    lines = [line.rstrip() for line in text.splitlines()]

    cleaned_blocks: list[str] = []
    paragraph_buffer: list[str] = []

    def flush_paragraph():
        if paragraph_buffer:
            # Join with a single space; trim extra spaces before punctuation
            joined = " ".join(paragraph_buffer).strip()
            joined = re.sub(r"\s+([.,;:])", r"\1", joined)  # fix "HIGGINS ."
            cleaned_blocks.append(joined)
            paragraph_buffer.clear()

    for raw_line in lines:
        # Normalize per-line and strip
        line = unicodedata.normalize("NFKC", raw_line).replace("\u00A0", " ").strip()

        # Blank line => paragraph boundary
        if not line:
            flush_paragraph()
            continue

        # Remove metadata/noise
        if REMOVE_RE.match(line):
            continue

        # Insert paragraph breaks for SPEAKER prefixes appearing mid-line (with or without preceding whitespace)
        for pref in SPEAKER_PREFIXES:
            # Add newline before prefix even if glued to previous characters
            line = re.sub(fr"(?<!^)(?<!\n)({re.escape(pref)}\s)", r"\n\1", line)

        # Normalize speaker tails: "Mr. SMITH of Texas." -> "Mr. SMITH."
        line = re.sub(r"\bof [A-Z][a-z]+(?: [A-Z][a-z]+)*\.", ".", line)

        # If the entire line is a TITLE, start a new block
        if TITLE_RE.fullmatch(line):
            flush_paragraph()
            cleaned_blocks.append("")   # hard boundary between speeches
            cleaned_blocks.append(line)
            continue

        # If line starts with a speaker prefix, start a new paragraph
        if line.startswith(SPEAKER_PREFIXES):
            flush_paragraph()
            paragraph_buffer.append(line)
        else:
            paragraph_buffer.append(line)

    # Flush any trailing paragraph
    flush_paragraph()

    # Final formatting: blank line after titles and after each block for readability
    output_lines: list[str] = []
    for block in cleaned_blocks:
        if TITLE_RE.fullmatch(block):
            output_lines.append(block)
            output_lines.append("")     # blank line after title
        else:
            output_lines.append(block)
            output_lines.append("")

    return "\n".join(output_lines).strip()


# ===============================
# MAIN LOOP
# ===============================

def main():
    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith(".txt"):
            continue

        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f"us_{filename}")

        with open(input_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        cleaned = clean_file(raw_text)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

    print("✅ Cleaning complete — output is hierarchical-chunk ready.")


if __name__ == "__main__":
    main()
