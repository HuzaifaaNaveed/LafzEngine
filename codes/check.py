import json
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

INPUTS_FILE = os.path.join(PARENT_DIR, "inputs.txt")
OUTPUTS_FILE = os.path.join(PARENT_DIR, "outputs.txt")
LOGS_FILE = os.path.join(PARENT_DIR, "logs.json")

URDUTOROMAN = {
    "ا": ["a", "aa", "", "i", "e", "u", "o",""],
    "آ": ["aa", "a", "ah"],
    "ب": ["b"],
    "پ": ["p"],
    "ت": ["t"],
    "ٹ": ["tt", "t", "ṭ"],
    "ث": ["s"],
    "ج": ["j"],
    "چ": ["ch", "c"],
    "ح": ["h", "ah", "eh", "uh"],
    "خ": ["kh", "x", "k"],
    "د": ["d"],
    "ڈ": ["dd", "d", "ḍ"],
    "ذ": ["z", "dh", "d"],
    "ر": ["r", "rr"],
    "ڑ": ["rr", "r", "ṛ", "rh"],
    "ز": ["z","s"],
    "ژ": ["zh", "z", "j"],
    "س": ["s", "c", "ss","sc"],
    "ش": ["sh", "s", "ch"],
    "ص": ["s", "ss"],
    "ض": ["z", "d", "dh"],
    "ط": ["t", "tt"],
    "ظ": ["z", "dh"],
    "ع": ["a", "e", "", "i", "u", "o", "aa"],
    "غ": ["gh", "g"],
    "ف": ["f", "ph"],
    "ق": ["q", "k"],
    "ک": ["k", "c", "q"],
    "گ": ["g"],
    "ل": ["l"],
    "م": ["m"],
    "ن": ["n"],
    "و": ["w", "v", "o", "oo", "u", "ou", "ow"],
    "ہ": ["h", "a", "e", "ah", "eh", "uh", "ha", "he"],
    "ھ": ["h", "hh"],
    "ء": ["", "a", "e", "i"],
    "ی": ["y", "i", "ee", "e", "ai", "ei",""],
    "ے": ["e", "ay", "ai", "a", "ye"],
    "أ": ["a", "", "aa"],
    "ئ": ["", "y", "i", "e"],
    "ؤ": ["w", "o", "", "u"],
}

URDUDIACRITICS = {
    "َ": ["a", "e", "u"],
    "ِ": ["i", "e", "y"], 
    "ُ": ["u", "o", "w"],
    "ْ": [""],
    "ّ": [""],
    "ٰ": ["a", "aa"],
    "ۡ": [""],
    "ٓ": ["aa", "a"],
    "۟": [""],
}

def checkTransliteration(urduText, romanOutput):
    romanLower = romanOutput.lower()
    romanWords = romanLower.split()
    
    for urduChar in urduText:
        if urduChar in URDUTOROMAN:
            possibleRomans = URDUTOROMAN[urduChar]
            charFound = False
            
            for possible in possibleRomans:
                if possible:
                    for word in romanWords:
                        if possible in word:
                            charFound = True
                            break
                if charFound:
                    break
                    
            if not charFound and urduChar not in URDUDIACRITICS:
                return False, f"'{urduChar}' not matched"
    
    return True, "Valid"

def read_files():
    with open(INPUTS_FILE, 'r', encoding='utf-8') as f:
        inputs = [line.strip() for line in f if line.strip()]
    
    with open(OUTPUTS_FILE, 'r', encoding='utf-8') as f:
        outputs = [line.strip() for line in f if line.strip()]
    
    return inputs, outputs

def clear_files():
    open(INPUTS_FILE, 'w', encoding='utf-8').close()
    open(OUTPUTS_FILE, 'w', encoding='utf-8').close()

def load_existing_logs():
    if os.path.exists(LOGS_FILE):
        with open(LOGS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_logs(logs):
    with open(LOGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def main():
    inputs, outputs = read_files()
    
    if len(inputs) != len(outputs):
        print("Error: inputs.txt and outputs.txt have different number of lines")
        return
    
    if not inputs:
        print("No data found in input files")
        return
    
    print(f"Validating {len(inputs)} transliterations...")
    
    existing_logs = load_existing_logs()
    new_logs = []
    
    for i, (urdu_input, model_output) in enumerate(zip(inputs, outputs)):
        print(f"Processing {i+1}/{len(inputs)}...")
        
        is_valid, message = checkTransliteration(urdu_input, model_output)
        verdict = "Valid" if is_valid else f"Invalid: {message}"
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": urdu_input,
            "output": model_output,
            "verdict": verdict
        }
        
        new_logs.append(log_entry)
        print(f"  Input: {urdu_input}")
        print(f"  Output: {model_output}")
        print(f"  Verdict: {verdict}")
        print()
    
    all_logs = existing_logs + new_logs
    save_logs(all_logs)
    
    clear_files()
    
    valid_count = sum(1 for log in new_logs if log['verdict'] == 'Valid')
    invalid_count = len(new_logs) - valid_count
    
    print(f"Validation complete!")
    print(f"Valid: {valid_count}")
    print(f"Invalid: {invalid_count}")
    print(f"Total logs in file: {len(all_logs)}")
    print(f"Logs saved to logs.json")
    print(f"Input and output files cleared")

if __name__ == '__main__':
    main()
