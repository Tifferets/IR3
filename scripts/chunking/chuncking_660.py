import nltk

# Make sure the Punkt tokenizer is available
nltk.download('punkt')
# pip install nltk


###############################################
# Utility functions
###############################################

def split_to_sentences(text):
    """
    Split the text into sentences using NLTK's sentence tokenizer.
    Decision: We use a real sentence splitter because sentence boundaries
    must be accurate — we are NOT allowed to split sentences in the middle.
    """
    return nltk.sent_tokenize(text)


def count_words(sentence):
    """
    Count words in a sentence.
    Decision: We count words using simple splitting by whitespace since
    chunk size is defined by word count (not tokens, not characters).
    """
    return len(sentence.split())


###############################################
# Main chunking function (Method 1)
###############################################

def chunk_fixed_overlap(text, max_words_per_chunk=660, overlap_sentences=3):
    """
    Create chunks of up to 660 words.
    Each new chunk starts with the last 3 sentences from the previous chunk.

    Variables:
        max_words_per_chunk  – strict word limit
        overlap_sentences    – number of sentences to repeat between chunks
    """

    sentences = split_to_sentences(text)
    chunks = []
    i = 0  # pointer over the sentences list

    # -----------------------------
    # Build the FIRST chunk
    # -----------------------------
    current_chunk = []
    current_word_count = 0

    while i < len(sentences):
        s = sentences[i]
        w = count_words(s)

        # Decision:
        # If adding this sentence would exceed 660 words → stop.
        if current_word_count + w <= max_words_per_chunk:
            current_chunk.append(s)
            current_word_count += w
            i += 1
        else:
            # If a single sentence is itself >660 words → it becomes a lone chunk.
            if current_word_count == 0:
                current_chunk = [s]
                i += 1
            break

    chunks.append(current_chunk)

    # -----------------------------
    # Build all NEXT chunks
    # -----------------------------
    while i < len(sentences):

        # Decision:
        # Start the new chunk with the last 3 sentences of the previous chunk.
        prev_chunk = chunks[-1]
        overlap = prev_chunk[-overlap_sentences:] if len(prev_chunk) >= overlap_sentences else prev_chunk

        current_chunk = overlap.copy()
        current_word_count = sum(count_words(s) for s in current_chunk)

        # Now keep adding new sentences until reaching 660 words
        while i < len(sentences):
            s = sentences[i]
            w = count_words(s)

            if current_word_count + w <= max_words_per_chunk:
                current_chunk.append(s)
                current_word_count += w
                i += 1
            else:
                # Same special-case logic if a huge sentence >660 words appears.
                if current_word_count == 0:
                    current_chunk = [s]
                    i += 1
                break

        chunks.append(current_chunk)

    return chunks


import os

if __name__ == "__main__":
    input_folder = "allData"
    output_folder = "chunks_output"

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in allData/
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".txt", ".md")):
            # Skip non-text files
            continue

        file_path = os.path.join(input_folder, filename)

        # Read file
        with open(file_path, "r", encoding="utf8") as f:
            text = f.read()

        # Chunk the document
        chunks = chunk_fixed_overlap(text)

        print(f"{filename}: {len(chunks)} chunks created")

        # Create a folder for this document
        file_output_dir = os.path.join(output_folder, filename + "_chunks")
        os.makedirs(file_output_dir, exist_ok=True)

        # Save chunks
        for idx, chunk in enumerate(chunks):
            chunk_text = "\n".join(chunk)
            chunk_filename = f"chunk_{idx+1}.txt"
            chunk_path = os.path.join(file_output_dir, chunk_filename)

            with open(chunk_path, "w", encoding="utf8") as out:
                out.write(chunk_text)
