import os
import xml.etree.ElementTree as ET
import re
from pathlib import Path
from tqdm import tqdm

# --- הגדרות קבועות ---
HIERARCHICAL_SEPARATOR = "\n\n" 
# * התיקון הקריטי כאן: טיפוס 2 רמות מעלה בהיררכיה *
# אם הסקריפט נמצא בתיקיית משנה ספציפית, זה מביא אותנו לבסיס הפרויקט.
BASE_DIR = Path(__file__).resolve().parent.parent

# --- פונקציות עזר קריטיות (הגדרות) ---

def clean_extracted_text(text):
    """מנקה טקסט: מסיר רווחים כפולים, שורות חדשות שאינן \n\n."""
    if not text:
        return ""
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_element(element):
    """חילוץ טקסט מתוך אלמנט כולל כל הטקסטים והזנבות (Tails) שלו באופן רקורסיבי."""
    text_parts = []
    if element.text:
        text_parts.append(clean_extracted_text(element.text))
        
    for child in element:
        if child.text:
            text_parts.append(clean_extracted_text(child.text))
        if child.tail:
            text_parts.append(clean_extracted_text(child.tail))
            
    return ' '.join(text_parts).strip()

def extract_hierarchical_text(xml_file_path):
    """מחלץ טקסט תוך שימור המבנה ההיררכי ע"י הפרדה באמצעות \n\n."""
    all_content = []
    
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        for child in root:
            tag = clean_extracted_text(child.tag)
            content_to_add = []

            # טיפול בכותרות
            if 'heading' in tag:
                heading_text = extract_text_from_element(child)
                if heading_text:
                    content_to_add.append(f"=== HEADING: {heading_text} ===")
            
            # טיפול בנאומים
            elif tag == 'speech':
                speech_paragraphs = []
                speaker_name = child.get('speakername', 'UNKNOWN SPEAKER')
                paragraphs = child.findall('p')
                
                for p_element in paragraphs:
                    paragraph_text = extract_text_from_element(p_element)
                    if paragraph_text:
                        paragraph_text = re.sub(r'\s+', ' ', paragraph_text).strip() 
                        if paragraph_text:
                             speech_paragraphs.append(paragraph_text)
                
                if speech_paragraphs:
                    speech_block = f"[{speaker_name}]:" 
                    speech_block += HIERARCHICAL_SEPARATOR + HIERARCHICAL_SEPARATOR.join(speech_paragraphs)
                    content_to_add.append(speech_block)

            if content_to_add:
                all_content.extend(content_to_add)

        final_text = HIERARCHICAL_SEPARATOR.join(all_content).strip()
        final_text = re.sub(r'(\n[ \t]*){2,}', HIERARCHICAL_SEPARATOR, final_text).strip()

        return final_text

    except ET.ParseError as e:
        print(f"⚠️ XML Parse Error in {Path(xml_file_path).name}: {e}")
        return ""
    except Exception as e:
        print(f"❌ General Error in {Path(xml_file_path).name}: {e}")
        return ""

def save_text_to_file(folder_path, filename, content):
    """שומר את הטקסט החולץ לקובץ יחיד בתיקייה שצוינה."""
    output_path = Path(folder_path)
    output_path.mkdir(parents=True, exist_ok=True) 
    
    output_file = output_path / filename
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"❌ שגיאה בשמירת הקובץ {filename}: {e}")

# --- הפונקציה המעודכנת להרצה על כל הקבצים ---
def process_all_files(input_folder_name="allData", 
                      output_folder_name="extracted_hierarchical_chunks",
                      file_prefix="UK"):
    
    # הנתיב כעת מחושב יחסית ל-BASE_DIR החדש (2 רמות למעלה)
    input_path = BASE_DIR / input_folder_name
    output_path = BASE_DIR / output_folder_name
    
    output_path.mkdir(parents=True, exist_ok=True) 

    xml_files = list(input_path.glob('*.xml'))
    
    if not input_path.exists():
        print(f"\n❌ שגיאה: תיקיית הקלט '{input_folder_name}' לא נמצאה בנתיב: {input_path}")
        print("ודאי שתיקיית allData נמצאת בנתיב הנכון ביחס לבסיס הפרויקט.")
        return

    if not xml_files:
        print(f"⚠️ לא נמצאו קבצי XML בתיקייה: {input_path}")
        return

    print(f"\n{'='*70}")
    print(f"מתחיל חילוץ של {len(xml_files)} קבצים מ: {input_path.resolve().name}")
    print(f"תחילית הקובץ שנקבעה: '{file_prefix}'")
    print(f"שומר פלט לתיקייה: {output_path.resolve().name}")
    print(f"{'='*70}")

    for xml_file_path in tqdm(xml_files, desc="Processing XML files"):
        
        text_content = extract_hierarchical_text(xml_file_path) 
        
        if text_content:
            original_stem = xml_file_path.stem
            # הפלט: UK_debates2023-06-28a.txt
            output_file_name = f"{file_prefix}_{original_stem}.txt" 
            
            save_text_to_file(output_path, output_file_name, text_content)

    print(f"\n✅ הסתיים חילוץ כל הקבצים. סה\"כ קבצים שנוצרו: {len(xml_files)}")
    print(f"הקבצים המנוקים נמצאים בתיקייה: {output_folder_name.upper()}")
    print("\n--- מוכן לשלב הבא: BM25 ו-Embeddings ---")


if __name__ == "__main__":
    process_all_files(
        input_folder_name="allData",
        output_folder_name="extracted_hierarchical_chunks",
        file_prefix="UK"
    )
