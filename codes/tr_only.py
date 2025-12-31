import torch
import pandas as pd
import time
import re
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from typing import List, Tuple, Dict

class UrduTransliterator:
    def __init__(self, model_path: str, csv_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_path)
        self.tokenizer.src_lang = "ur"
        self.tokenizer.tgt_lang = "en"
        
        self.model.to(self.device)
        self.model.eval()
        
        self.lookup_dict = self._load_csv_lookup(csv_path)
        print(f"Loaded {len(self.lookup_dict)} entries from CSV")
    
    def _load_csv_lookup(self, csv_path: str) -> Dict[str, str]:
        df = pd.read_csv(csv_path, encoding='utf-8')
        lookup_dict = {}
        
        for _, row in df.iterrows():
            urdu_text = str(row['input']).strip()
            transliterated = str(row['output']).strip()
            if urdu_text and transliterated:
                lookup_dict[urdu_text] = transliterated
        
        return lookup_dict
    
    def _preprocess_sentence(self, sentence: str) -> str:
        sentence = re.sub(r'،', ' ،', sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        return sentence.strip()
    
    def _tokenize_sentence(self, sentence: str) -> List[str]:
        parts = sentence.split()
        tokens = []
        
        for part in parts:
            if re.search(r'[،۔!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>/?]+$', part):
                word_match = re.match(r'^([^،۔!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>/?]+)([،۔!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>/?]+)$', part)
                if word_match:
                    word = word_match.group(1)
                    punctuation = word_match.group(2)
                    if word:
                        tokens.append(word)
                    tokens.append(punctuation)
                else:
                    tokens.append(part)
            else:
                tokens.append(part)
        
        return tokens
    
    def _find_sequences(self, tokens: List[str]) -> List[Tuple[int, int, str]]:
        sequences = []
        n = len(tokens)
        
        for start in range(n):
            if self._is_punctuation(tokens[start]):
                continue
                
            for end in range(start + 1, n + 1):
                sequence_tokens = tokens[start:end]
                if all(self._is_punctuation(token) for token in sequence_tokens):
                    continue
                
                clean_sequence_tokens = [token for token in sequence_tokens if not self._is_punctuation(token)]
                sequence_text = " ".join(clean_sequence_tokens)
                
                if sequence_text in self.lookup_dict:
                    sequences.append((start, end, sequence_text))
        
        return sequences
    
    def _is_punctuation(self, token: str) -> bool:
        return bool(re.match(r'^[،۔!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>/?]+$', token))
    
    def _find_non_overlapping_sequences(self, tokens: List[str]) -> List[Tuple[int, int, str]]:
        sequences = self._find_sequences(tokens)
        used_indices = set()
        selected_sequences = []
        
        sequences.sort(key=lambda x: ((x[1] - x[0]), -x[0]), reverse=True)
        
        for start, end, sequence_text in sequences:
            overlap = False
            sequence_indices = []
            
            for i in range(start, end):
                if not self._is_punctuation(tokens[i]):
                    sequence_indices.append(i)
                    if i in used_indices:
                        overlap = True
            
            if not overlap:
                selected_sequences.append((start, end, sequence_text))
                for i in sequence_indices:
                    used_indices.add(i)
        
        selected_sequences.sort(key=lambda x: x[0])
        return selected_sequences
    
    def _capitalize_first_letter(self, word: str) -> str:
        if word and len(word) > 0:
            if word[0].islower():
                return word[0].upper() + word[1:]
        return word
    
    def _merge_consecutive_capitals(self, sentence: str) -> str:
        words = sentence.split()
        result = []
        i = 0
        
        while i < len(words):
            word = words[i]
            
            if len(word) == 1 and word.isupper():
                consecutive_capitals = [word]
                j = i + 1
                
                while j < len(words) and len(words[j]) == 1 and words[j].isupper():
                    consecutive_capitals.append(words[j])
                    j += 1
                
                if len(consecutive_capitals) > 1:
                    result.append(''.join(consecutive_capitals))
                    i = j
                else:
                    result.append(word)
                    i += 1
            else:
                result.append(word)
                i += 1
        
        return ' '.join(result)
    
    def _fix_comma_spacing(self, sentence: str) -> str:
        sentence = re.sub(r'\s+,', ',', sentence)
        return sentence
    
    def transliterate_with_lookup(self, urdu_sentence: str) -> Tuple[str, Dict]:
        urdu_sentence = self._preprocess_sentence(urdu_sentence)
        
        tokens = self._tokenize_sentence(urdu_sentence)
        
        if not tokens:
            return "", {}
        
        sequences = self._find_non_overlapping_sequences(tokens)
        used_indices = set()
        transliteration_map = {}
        
        for start, end, sequence_text in sequences:
            transliteration = self.lookup_dict[sequence_text]
            
            sequence_tokens = []
            for i in range(start, end):
                if not self._is_punctuation(tokens[i]):
                    sequence_tokens.append(tokens[i])
                    used_indices.add(i)
            
            transliterated_parts = transliteration.split()
            if len(sequence_tokens) == len(transliterated_parts):
                for token, translit_part in zip(sequence_tokens, transliterated_parts):
                    token_indices = [j for j, t in enumerate(tokens) if t == token and j >= start and j < end]
                    for idx in token_indices:
                        transliteration_map[f"{token}_{idx}"] = {
                            'transliteration': translit_part,
                            'source': 'csv',
                            'indices': [idx],
                            'original_token': token
                        }
            else:
                transliteration_map[f"{sequence_text}_{start}"] = {
                    'transliteration': transliteration,
                    'source': 'csv',
                    'indices': list(range(start, end)),
                    'original_token': sequence_text
                }
        
        remaining_tokens = []
        remaining_indices = []
        for i, token in enumerate(tokens):
            if i not in used_indices:
                remaining_tokens.append(token)
                remaining_indices.append(i)
        
        tokens_to_transliterate = []
        token_indices = []
        
        for i, token in enumerate(remaining_tokens):
            if not self._is_punctuation(token):
                tokens_to_transliterate.append(token)
                token_indices.append(remaining_indices[i])
        
        model_transliterations = []
        if tokens_to_transliterate:
            model_transliterations = self.transliterate_batch(tokens_to_transliterate)
            
        translit_index = 0
        for i, token in enumerate(remaining_tokens):
            original_index = remaining_indices[i]
            if self._is_punctuation(token):
                if token == '،':
                    english_punct = ','
                else:
                    english_punct = token
                transliteration_map[f"{token}_{original_index}"] = {
                    'transliteration': english_punct,
                    'source': 'punctuation',
                    'indices': [original_index],
                    'original_token': token
                }
            else:
                transliteration_map[f"{token}_{original_index}"] = {
                    'transliteration': model_transliterations[translit_index],
                    'source': 'model',
                    'indices': [original_index],
                    'original_token': token
                }
                translit_index += 1
        
        final_transliteration = []
        for i in range(len(tokens)):
            for key, info in transliteration_map.items():
                if i in info['indices'] and i == info['indices'][0]:
                    final_transliteration.append(info['transliteration'])
                    break
        
        result = " ".join(final_transliteration)
        result = self._fix_comma_spacing(result)
        
        words = result.split()
        capitalized_words = []
        for word in words:
            if word and not re.match(r'^[^a-zA-Z]+$', word):
                if re.search(r'[a-zA-Z]', word):
                    capitalized_words.append(self._capitalize_first_letter(word))
                else:
                    capitalized_words.append(word)
            else:
                capitalized_words.append(word)
        
        result = ' '.join(capitalized_words)
        result = self._merge_consecutive_capitals(result)
        
        display_map = {}
        for key, info in transliteration_map.items():
            original = info['original_token']
            display_map[original] = {
                'transliteration': info['transliteration'],
                'source': info['source'],
                'indices': info['indices']
            }
        
        return result, display_map
    
    def transliterate_batch(self, urdu_texts: List[str]):
        inputs = self.tokenizer(
            urdu_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=2,
                early_stopping=True
            )
        
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

def read_inputs(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

def write_outputs(file_path: str, outputs: list):
    with open(file_path, 'w', encoding='utf-8') as file:
        for output in outputs:
            file.write(output + '\n')

def main():
    model_path = r"C:\Users\zeffn\OneDrive\Desktop\transliterationPipeline\model"
    csv_path = r"C:\Users\zeffn\OneDrive\Desktop\transliterationPipeline\db\tr_v8.csv"
    
    transliterator = UrduTransliterator(model_path, csv_path)
    
    input_texts = read_inputs(r"C:\Users\zeffn\OneDrive\Desktop\transliterationPipeline\inputs.txt")
    
    if not input_texts:
        print("No inputs found in inputs.txt")
        return
    
    print(f"Transliterating {len(input_texts)} inputs...")
    print("-" * 80)
    
    total_time = 0
    roman_transliterations = []
    detailed_results = []
    
    for i, sentence in enumerate(input_texts):
        print(f"Processing sentence {i+1}/{len(input_texts)}: {sentence}")
        
        start_time = time.time()
        transliterated_sentence, details = transliterator.transliterate_with_lookup(sentence)
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000
        total_time += response_time
        
        roman_transliterations.append(transliterated_sentence)
        detailed_results.append({
            'input': sentence,
            'output': transliterated_sentence,
            'response_time_ms': response_time,
            'details': details
        })
        
        print(f"  Response time: {response_time:.2f} ms")
        print(f"  Final output: {transliterated_sentence}")
        print("  Breakdown:")
        for seq, info in details.items():
            source = info['source']
            translit = info['transliteration']
            indices = info['indices']
            print(f"    '{seq}' -> '{translit}' ({source}) [indices: {indices}]")
        print("-" * 80)
    
    write_outputs(r"C:\Users\zeffn\OneDrive\Desktop\transliterationPipeline\outputs.txt", roman_transliterations)
    
    avg_time = total_time / len(input_texts)
    
    print("\n" + "=" * 80)
    print("TRANSLITERATION COMPLETED!")
    print("=" * 80)
    print(f"Total sentences processed: {len(input_texts)}")
    print(f"Total time: {total_time:.2f} ms")
    print(f"Average response time: {avg_time:.2f} ms per sentence")
    print("\nDetailed Results:")
    
    for i, result in enumerate(detailed_results):
        print(f"{i+1}. Input: {result['input']}")
        print(f"   Output: {result['output']}")
        print(f"   Time: {result['response_time_ms']:.2f} ms")
        
        csv_count = sum(1 for info in result['details'].values() if info['source'] == 'csv')
        model_count = sum(1 for info in result['details'].values() if info['source'] == 'model')
        punct_count = sum(1 for info in result['details'].values() if info['source'] == 'punctuation')
        print(f"   Sources: CSV={csv_count}, Model={model_count}, Punctuation={punct_count}")
        print()

if __name__ == "__main__":
    main()