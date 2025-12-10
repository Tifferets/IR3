import os
import re
import html
from pathlib import Path

# ------------------------------------
# CONFIG
# ------------------------------------
INPUT_FOLDER = "US_congressional_speeches_Text_Files"
OUTPUT_FOLDER = "cleanedData"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ------------------------------------
# CLEANING FUNCTION
# ------------------------------------

def clean_text(text):
    """
    Cleaning for hierarchical chunking.
    Keeps:
        - stopwords
        - punctuation
        - capitalization
        - paragraph structure
    Removes:
        - Volume / Issue / Pages metadata
        - Section header
        - ===== separator lines
        - <pre> blocks
        - [Page E635]
        - [Extensions of Remarks]
        - From the Congressional Record Online...
        - HTML escape symbols (&#x27; etc.)
        - HTML links <a href="...">
        - Lone ______ lines
    """

    # Decode HTML escape codes
    text = html.unescape(text)

    # Remove metadata sections
    text = re.sub(r"Volume:\s*\d+.*?\n", "", text)
    text = re.sub(r"Pages?:\s*[A-Z0-9\- ]+\n", "", text)
    text = re.sub(r"Section:.*?\n", "", text)
    text = re.sub(r"Date:.*?\n", "", text)

    # Remove heavy separators
    text = re.sub(r"=+", "", text)

    # Remove <pre> wrappers
    text = re.sub(r"<pre>|</pre>", "", text)

    # Remove page markers
    text = re.sub(r"\[Extensions of Remarks\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Page [A-Z0-9\-]+\]", "", text)
    text = re.sub(r"\[\[Page [A-Z0-9\-]+\]\]", "", text)

    # Remove congressional boilerplate
    text = re.sub(r"From the Congressional Record Online.*?\n", "", text)

    # Remove HTML anchors but keep visible text
    text = re.sub(r"<a[^>]*>(.*?)</a>", r"\1", text)

    # Remove decorative line ______
    text = re.sub(r"^_{3,}$", "", text, flags=re.MULTILINE)

    # Fix “one of Newsweek&#x27;s Best”
    text = text.replace("Newsweek&#x27;s", "Newsweek’s")
    text = text.replace("Doctor&#x27;s", "Doctor’s")

    # Collapse excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize newlines but preserve paragraph structure
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ------------------------------------
# PROCESS ONLY US FILES
# ------------------------------------

for filename in os.listdir(INPUT_FOLDER):

    # Only process files beginning with "US_"
    if not filename.startswith("US_"):
        print(f"SKIPPED (not US): {filename}")
        continue

    # Only text formats
    if not filename.lower().endswith(".txt"):
        print(f"SKIPPED (not .txt): {filename}")
        continue

    input_path = Path(INPUT_FOLDER) / filename
    output_path = Path(OUTPUT_FOLDER) / filename

    print(f"Cleaning: {filename}")

    raw = input_path.read_text(encoding="utf8")
    cleaned = clean_text(raw)
    output_path.write_text(cleaned, encoding="utf8")

print("✓ CLEANING DONE (US files only)")
