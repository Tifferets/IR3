import os
import re
from pathlib import Path
from nltk.corpus import stopwords
import nltk
import shutil 

# -------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------
# מניח שהקובץ הזה נמצא בתוך תיקיית scripts
BASE_DIR = Path(__file__).resolve().parent 
ROOT_DIR = BASE_DIR.parent 

# תיקיית הקבצים המאוחדים (מקור) - מניח שזו התיקייה שנוצרה ב-Stage 1
INPUT_FOLDER = ROOT_DIR / "allData" 
# תיקיית יעד חדשה לקבצים המנוקים חזק (Output)
OUTPUT_FOLDER = ROOT_DIR / "allData_super_cleaned" 

# יצירת תיקיית יעד אם אינה קיימת
OUTPUT_FOLDER.mkdir(exist_ok=True)


# -------------------------------------------------------------
# PRE-REQUISITES (הבטחת הורדת משאבי NLTK)
# -------------------------------------------------------------

# ננסה להוריד את משאב stopwords אם הוא חסר
try:
    nltk.data.find('corpora/stopwords')
except LookupError: 
    print("Downloading nltk stopwords resource...")
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Warning: Failed to download nltk stopwords, may affect cleanup quality: {e}")


# -------------------------------------------------------------
# רשימות מילים להסרה
# -------------------------------------------------------------
# 1. מילות עצירה סטנדרטיות באנגלית (נבחרות מ-NLTK)
ENGLISH_STOPWORDS = set(stopwords.words('english'))

# 2. מילים חושפניות ספציפיות לפרויקט (שמגלים את הקלאס)
REVEALING_WORDS = {
    'uk', 'us', 'usa', 'united', 'kingdom', 'states', 'britain', 
    'america', 'congress', 'parliament', 
    # --- הוספות חדשות לטיפול בגרש שהוסר ---
    'uks', 'uss', 
    # מילים ותארים ששרדו ניקוי
    'mr', 'ms', 'mrs', 'speaker', 'hon', 'honorable', 'sir', 'doctor', 'deputy', 'superintendent', 'charles'
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
    # 1. המרה לאותיות קטנות
    text = text.lower()
    
    # 2. הסרת סימני פיסוק וספרות
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 3. פיצול לטוקנים, סינון מילים, וחיבור מחדש
    tokens = text.split()
    
    # סינון: שמור רק מילים שאינן ברשימת ההסרה ושאינן קצרות מדי (לפחות 2 תווים)
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
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
print(f"\n✅ Enhanced cleanup complete. {processed_count} files processed and saved to {OUTPUT_FOLDER.name}.")