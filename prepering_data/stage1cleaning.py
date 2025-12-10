import os
import re
import html

INPUT_FOLDER = "US_congressional_speeches_Text_Files"
OUTPUT_FOLDER = "cleanedData_us"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def clean_text(text):
    # Fix HTML escape codes (&#x27; → ')
    text = html.unescape(text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip BOM if present
    text = text.lstrip("\ufeff")

    # -----------------------------------------
    # REMOVE METADATA LINES ONLY
    # -----------------------------------------
    meta_patterns = [
        r"^\s*Title:.*$",
        r"^\s*Volume:.*$",
        r"^\s*Issue:.*$",
        r"^\s*Pages?:.*$",
        r"^\s*Section:.*$",
        r"^\s*Date:.*$",
        r"^\s*={5,}\s*$",
    ]

    for p in meta_patterns:
        text = re.sub(p, "", text, flags=re.MULTILINE)

    # Remove <pre> tags
    text = re.sub(r"</?pre>", "", text)

    # -----------------------------------------
    # REMOVE PAGE HEADERS
    # -----------------------------------------
    page_patterns = [
        r"^\s*\[Page [A-Z0-9\-\s]+\]\s*$",
        r"^\s*\[\[Page [A-Z0-9\-\s]+\]\]\s*$",
        r"^\s*\[Pages? [A-Z0-9\-\s]+\]\s*$",
        r"^\s*\[Extensions of Remarks\]\s*$",
    ]
    for p in page_patterns:
        text = re.sub(p, "", text, flags=re.MULTILINE)

    # -----------------------------------------
    # REMOVE BOILERPLATE BLOCKS
    # -----------------------------------------
    text = re.sub(
        r"^\s*From the Congressional Record Online.*$",
        "",
        text,
        flags=re.MULTILINE
    )

    # Remove lines of only underscores (with spaces)
    text = re.sub(r"^\s*_{3,}\s*$", "", text, flags=re.MULTILINE)

    # Remove HTML link brackets
    text = re.sub(r"\[<.*?>\]", "", text)

    # -----------------------------------------
    # CLEAN UP EXTRA SPACE
    # -----------------------------------------
    text = re.sub(r"\n{3,}", "\n\n", text)   # Collapse blank lines
    text = re.sub(r"[ \t]+", " ", text)      # Clean spaces
    text = text.strip()

    return text


# -----------------------------------------
# MAIN LOGIC — process ONLY US_* files
# -----------------------------------------
for filename in os.listdir(INPUT_FOLDER):

    if not filename.lower().endswith((".txt", ".md")):
        continue

    input_path = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    print("Cleaning:", filename)

    with open(input_path, "r", encoding="utf8") as f:
        raw = f.read()

    cleaned = clean_text(raw)

    with open(output_path, "w", encoding="utf8") as f:
        f.write(cleaned)

print("✓ CLEANING DONE")
