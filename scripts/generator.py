
import requests
import time
import sys
import json

# הגדרת קידוד למניעת שגיאות בטרמינל
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# רשימת המפתחות שלך
API_KEYS = [
    "AIzaSyCamchq0QuZE_9tx0n28c7bgUJlQwgfOL4",
    "AIzaSyAncD2JWXV7SQHGaiJlZCrlqf3yxrHn3Ew"
]
current_key_index = 0

def get_next_key():
    """מחליף בין המפתחות ברשימה"""
    global current_key_index
    key = API_KEYS[current_key_index % len(API_KEYS)]
    current_key_index += 1
    return key

def generate_answer(query, retrieved_chunks):
    """
    מייצרת תשובה על סמך צ'אנקים בלבד (אחריות אדם ב').
    משתמשת ברוטציה בין מפתחות ובנתיב Gemini 2.5 Flash.
    """
    # 1. בניית הקונטקסט
    context_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        context_text += f"\n--- Source File: {chunk['file_name']} ---\n"
        context_text += chunk['text'] + "\n"

    # 2. הנחיות המערכת של אדם ב'
    system_prompt = (
        "You are an expert information extraction assistant. "
        "Your task is to answer the user's question based ONLY on the provided document chunks. "
        "CRITICAL RULES:\n"
        "1. Zero External Knowledge: Do not use any information that is not explicitly mentioned in the provided chunks.\n"
        "2. Handling Missing Info: If the provided chunks do not contain enough information to answer the question, "
        "state: 'I cannot answer this question based on the provided documents'.\n"
        "3. Source Attribution: For every fact you state, you must mention the source file name.\n"
        "4. Handling Noise: If some chunks are irrelevant to the query, ignore them.\n"
        "5. Consistency: If different chunks provide conflicting information, report both views."
    )

    # 3. מנגנון הרצה עם רוטציה והמתנה
    retries = 6 
    for i in range(retries):
        current_key = get_next_key()
        # שימוש בנתיב המדויק שביקשת
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={current_key}"

        payload = {
            "contents": [{"parts": [{"text": f"Context:\n{context_text}\n\nQuestion: {query}"}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {"temperature": 0} # דיוק עובדתי
        }

        try:
            response = requests.post(url, json=payload)

            if response.status_code == 200:
                result = response.json()
                return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "No response.")

            if response.status_code == 429:
                wait_time = 2**i
                print(f"Key {current_key_index % len(API_KEYS)} limited. Rotating and waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"Error {response.status_code} with key {current_key_index % len(API_KEYS)}. Trying next key...")
                continue

        except Exception as e:
            if i == retries - 1:
                return f"Failed after multiple keys/retries: {str(e)}"
            time.sleep(1)

    return "Failed to get response after rotating all keys and maximum retries."

if __name__ == "__main__":
    mock_chunks = [{"text": "Speech on July 3rd about defense budget.", "file_name": "uk_2023-07-03.txt"}]
    test_query = "What was the speech about?"
    print("--- Running Dual-Key Rotation Test ---")
    print(generate_answer(test_query, mock_chunks))