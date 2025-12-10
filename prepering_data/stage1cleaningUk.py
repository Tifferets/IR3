import re
from pathlib import Path
from tqdm import tqdm

# --- CONSTANTS ---

HIERARCHICAL_SEPARATOR = "\n\n"

# BASE_DIR = main project folder (IR3)
BASE_DIR = Path(__file__).resolve().parent.parent


# --- TEXT CLEANING FUNCTIONS ---

def clean_extracted_text(text: str) -> str:
    """
    Cleans raw text by:
    - removing tabs and carriage returns,
    - collapsing multiple spaces into a single space.
    """
    if not text:
        return ""
    text = re.sub(r'[\r\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_hierarchical_text(txt_file_path: Path) -> str:
    """
    Reads a plain text file and returns a cleaned version of its content.
    """
    try:
        with open(txt_file_path, "r", encoding="utf-8") as f:
            text = f.read()

        text = clean_extracted_text(text)

        # Reduce multiple blank lines to max 2
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    except Exception as e:
        print(f"❌ Error reading file {txt_file_path.name}: {e}")
        return ""


def save_text_to_file(folder_path: Path, filename: str, content: str) -> None:
    """
    Saves cleaned text to a specified output file.
    """
    folder_path.mkdir(parents=True, exist_ok=True)
    output_file = folder_path / filename

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"❌ Error saving file {filename}: {e}")


# --- MAIN PROCESSING FUNCTION ---

def process_all_files(
    input_folder_name: str = "UK_british_debates_text_files_normalize",
    output_folder_name: str = "cleanedData_uk",
    file_prefix: str = "UK",
) -> None:
    """
    Processes all TXT files inside the input folder:
    - reads them
    - cleans text
    - saves output into a dedicated cleanedData_uk folder
    """

    input_path = BASE_DIR / input_folder_name
    output_path = BASE_DIR / output_folder_name

    # Validate input folder
    if not input_path.exists():
        print(f"\n❌ ERROR: Input folder '{input_folder_name}' not found at: {input_path}")
        print("Please verify the folder path relative to BASE_DIR.")
        return

    txt_files = sorted(input_path.glob("*.txt"))

    if not txt_files:
        print(f"⚠️ WARNING: No TXT files found in: {input_path}")
        return

    print(f"\n{'=' * 70}")
    print(f"Starting text cleaning for {len(txt_files)} files")
    print(f"Input folder : {input_path.resolve()}")
    print(f"Output folder: {output_path.resolve()}")
    print(f"File prefix  : '{file_prefix}'")
    print(f"{'=' * 70}")

    for file_path in tqdm(txt_files, desc="Processing UK text files"):
        text_content = extract_hierarchical_text(file_path)

        if text_content:
            original_stem = file_path.stem  # e.g., debates2023-06-28

            # Avoid double-UK prefix
            if original_stem.startswith(file_prefix + "_"):
                output_filename = f"{original_stem}.txt"
            else:
                output_filename = f"{file_prefix}_{original_stem}.txt"

            save_text_to_file(output_path, output_filename, text_content)

    print(f"\n✅ DONE! Total processed files: {len(txt_files)}")
    print(f"Cleaned files saved to: {output_folder_name}")
    print("\n--- Ready for next stage: BM25 + Embeddings ---")


# --- SCRIPT ENTRY POINT ---

if __name__ == "__main__":
    process_all_files(
        input_folder_name="UK_british_debates_text_files_normalize",
        output_folder_name="cleanedData_uk",
        file_prefix="UK",
    )
