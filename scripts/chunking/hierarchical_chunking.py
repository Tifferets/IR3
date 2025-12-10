import os
import re
import spacy

nlp = spacy.load("en_core_web_sm")


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def split_sentences(text):
    """Split into sentences using spaCy."""
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def clean_line(line):
    """Normalize spacing."""
    return line.strip()


# --------------------------------------------------
# Hierarchical Chunking
# --------------------------------------------------

def hierarchical_chunk(text):
    """
    1. Split by SECTION headers (very common in Congress / Bills / Reports)
    2. Inside each section, split by paragraphs
    3. Inside each paragraph, split into sentences
    Return a list of chunk strings.
    """

    # 1. SECTION SPLIT
    section_regex = r"(Section\s+\d+[\.:]?|Sec\.\s*\d+[\.:]?)"
    sections = re.split(section_regex, text, flags=re.IGNORECASE)

    chunks = []
    buffer = ""

    for part in sections:
        part = part.strip()
        if not part:
            continue

        # 2. PARAGRAPH SPLIT
        paragraphs = [clean_line(p) for p in part.split("\n\n") if clean_line(p)]

        for para in paragraphs:
            # 3. Sentence split
            sentences = split_sentences(para)

            # Build chunk
            chunk_text = "\n".join(sentences)
            if chunk_text.strip():
                chunks.append(chunk_text)

    return chunks


# --------------------------------------------------
# Save each chunk as a .txt file
# --------------------------------------------------

def save_chunks(chunks, output_dir, base_filename):
    os.makedirs(output_dir, exist_ok=True)

    for idx, chunk in enumerate(chunks, start=1):
        out_path = os.path.join(output_dir, f"{base_filename}_chunk_{idx}.txt")
        with open(out_path, "w", encoding="utf8") as f:
            f.write(chunk)


# --------------------------------------------------
# Main Runner
# --------------------------------------------------

def run_chunker(input_folder="allData", output_folder="hierarchical_chunks"):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.endswith(".txt"):
            continue

        print(f"Processing: {filename}")

        file_path = os.path.join(input_folder, filename)
        with open(file_path, "r", encoding="utf8") as f:
            text = f.read()

        chunks = hierarchical_chunk(text)
        print(f"  → {len(chunks)} chunks created")

        save_chunks(
            chunks,
            output_dir=os.path.join(output_folder, filename + "_chunks"),
            base_filename=filename.replace(".txt", "")
        )


if __name__ == "__main__":
    run_chunker()
