import os
import spacy

###############################################
# Load optimized spaCy model
###############################################

nlp = spacy.load(
    "en_core_web_sm",
    disable=["ner", "parser", "tagger", "lemmatizer"]
)

# Add sentence splitter
nlp.add_pipe("sentencizer")

# Allow large files
nlp.max_length = 3_000_000



###############################################
# Utility functions
###############################################

def split_to_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def count_words(sentence):
    return len(sentence.split())


###############################################
# Chunking Method 1 (fixed size + overlap)
###############################################

def chunk_fixed_overlap(text, max_words_per_chunk=660, overlap_sentences=3):
    sentences = split_to_sentences(text)
    chunks = []
    i = 0

    # ----- FIRST CHUNK -----
    current_chunk = []
    current_word_count = 0

    while i < len(sentences):
        s = sentences[i]
        w = count_words(s)

        if current_word_count + w <= max_words_per_chunk:
            current_chunk.append(s)
            current_word_count += w
            i += 1
        else:
            if current_word_count == 0:
                current_chunk = [s]
                i += 1
            break

    chunks.append(current_chunk)

    # ----- NEXT CHUNKS -----
    while i < len(sentences):
        prev_chunk = chunks[-1]
        overlap = prev_chunk[-overlap_sentences:] if len(prev_chunk) >= overlap_sentences else prev_chunk

        current_chunk = overlap.copy()
        current_word_count = sum(count_words(s) for s in current_chunk)

        while i < len(sentences):
            s = sentences[i]
            w = count_words(s)

            if current_word_count + w <= max_words_per_chunk:
                current_chunk.append(s)
                current_word_count += w
                i += 1
            else:
                if current_word_count == 0:
                    current_chunk = [s]
                    i += 1
                break

        chunks.append(current_chunk)

    return chunks


###############################################
# Main processing loop
###############################################

if __name__ == "__main__":
    input_folder = "allData"
    output_folder = "chunks_output"

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".txt", ".md")):
            continue

        file_output_dir = os.path.join(output_folder, filename + "_chunks")

        # ------------------------------------------
        # SKIP IF ALREADY PROCESSED
        # ------------------------------------------
        if os.path.exists(file_output_dir) and len(os.listdir(file_output_dir)) > 0:
            print(f"Skipping (already done): {filename}")
            continue

        print("Processing:", filename)

        file_path = os.path.join(input_folder, filename)
        with open(file_path, "r", encoding="utf8") as f:
            text = f.read()

        chunks = chunk_fixed_overlap(text)
        print(f"  → {len(chunks)} chunks created")

        os.makedirs(file_output_dir, exist_ok=True)

        for idx, chunk in enumerate(chunks):
            chunk_text = "\n".join(chunk)
            chunk_filename = f"chunk_{idx+1}.txt"
            chunk_path = os.path.join(file_output_dir, chunk_filename)

            with open(chunk_path, "w", encoding="utf8") as out:
                out.write(chunk_text)

    print("\nAll done.")
