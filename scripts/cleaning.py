import os
import re
import shutil
from nltk.corpus import stopwords
import nltk
from pathlib import Path

# ודא שמשאבי NLTK הדרושים מותקנים
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("Downloading nltk stopwords...")
    nltk.download('stopwords', quiet=True)

# -------------------------------------------------------------
# הגדרות נתיבים
# -------------------------------------------------------------
# מניח שהקובץ הזה נמצא בתוך תיקיית scripts
BASE_DIR = Path(__file__).resolve().parent 
ROOT_DIR = BASE_DIR.parent # INFO_RETRIEVAL02

# תיקיית הקבצים המאוחדים (מקור)
INPUT_FOLDER = ROOT_DIR / "allData" 
# תיקיית יעד חדשה לקבצים המנוקים חזק
OUTPUT_FOLDER = ROOT_DIR / "allData_super_cleaned" 

# יצירת תיקיית יעד אם אינה קיימת
OUTPUT_FOLDER.mkdir(exist_ok=True)

# -------------------------------------------------------------
# רשימות מילים להסרה
# -------------------------------------------------------------
# 1. מילות עצירה סטנדרטיות באנגלית
ENGLISH_STOPWORDS = set(stopwords.words('english'))

# 2. מילים חושפניות ספציפיות לפרויקט (יש להוסיף את כל הוריאציות)
# הוספנו מילים שחוזרות על עצמן במערכת (כמו "Mr Speaker" שהיה יכול לחזור)
REVEALING_WORDS = {
    'uk', 'us', 'usa', 'united', 'kingdom', 'states', 'britain', 
    'america', 'congress', 'parliament', 
    # מילים ותארים שחושפים את המוסד (עשויים להופיע אחרי ניקוי Stage 1)
    'mr', 'ms', 'mrs', 'speaker', 'hon', 'honorable', 'sir'
}

# איחוד רשימות ההסרה
ALL_WORDS_TO_REMOVE = ENGLISH_STOPWORDS.union(REVEALING_WORDS)
print(f"Loaded {len(ALL_WORDS_TO_REMOVE)} total words for enhanced removal.")


# -------------------------------------------------------------
# פונקציית הניקוי החזק
# -------------------------------------------------------------
def perform_enhanced_cleanup(text):
    """
    מבצע ניקוי חזק: מוריד את כל מילות העצירה והמילים החושפניות.
    """
    # 1. המרה לאותיות קטנות (להבטחת עקביות בניקוי)
    text = text.lower()
    
    # 2. הסרת סימני פיסוק וספרות (משאיר רק אותיות ורווחים)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 3. פיצול לטוקנים, סינון מילים, וחיבור מחדש
    tokens = text.split()
    
    # סינון: שמור רק מילים שאינן ברשימת ההסרה
    filtered_tokens = [word for word in tokens if word not in ALL_WORDS_TO_REMOVE and len(word) > 1]
    
    # 4. חיבור הטקסט המנוקה
    return ' '.join(filtered_tokens)


# -------------------------------------------------------------
# עיבוד הקבצים
# -------------------------------------------------------------
print(f"\n=== Starting Enhanced Cleanup from {INPUT_FOLDER.name} to {OUTPUT_FOLDER.name} ===")
processed_count = 0

for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith('.txt'):
        input_path = INPUT_FOLDER / filename
        output_path = OUTPUT_FOLDER / filename
        
        try:
            # קריאת הקובץ
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
            
            # ניקוי חזק
            cleaned_text = perform_enhanced_cleanup(raw_text)
            
            # שמירה לקובץ החדש
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            # print(f"Cleaned and saved: {filename}")
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
print(f"\n✅ Enhanced cleanup complete. {processed_count} files processed.")