import torch
import time
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

class UrduTransliterator:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_path)
        self.tokenizer.src_lang = "ur"
        self.tokenizer.tgt_lang = "en"
        
        self.model.to(self.device)
        self.model.eval()
    
    def transliterate(self, urdu_sentence: str) -> str:
        # Tokenize the entire sentence
        inputs = self.tokenizer(
            urdu_sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        # Generate transliteration
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=2,
                early_stopping=True
            )
        
        # Decode the output
        transliterated = self.tokenizer.batch_decode(
            generated_tokens, 
            skip_special_tokens=True
        )
        
        return transliterated[0]

def read_inputs(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

def write_outputs(file_path: str, outputs: list):
    with open(file_path, 'w', encoding='utf-8') as file:
        for output in outputs:
            file.write(output + '\n')

def main():
    model_path = r"C:\Users\zeffn\OneDrive\Desktop\transliterationPipeline\model"
    
    transliterator = UrduTransliterator(model_path)
    
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
        transliterated_sentence = transliterator.transliterate(sentence)
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000
        total_time += response_time
        
        roman_transliterations.append(transliterated_sentence)
        detailed_results.append({
            'input': sentence,
            'output': transliterated_sentence,
            'response_time_ms': response_time
        })
        
        print(f"  Response time: {response_time:.2f} ms")
        print(f"  Output: {transliterated_sentence}")
        print("-" * 80)
    
    write_outputs(r"C:\Users\zeffn\OneDrive\Desktop\transliterationPipeline\outputs.txt", roman_transliterations)
    
    avg_time = total_time / len(input_texts)
    
    print("\n" + "=" * 80)
    print("TRANSLITERATION COMPLETED!")
    print("=" * 80)
    print(f"Total sentences processed: {len(input_texts)}")
    print(f"Total time: {total_time:.2f} ms")
    print(f"Average response time: {avg_time:.2f} ms per sentence")
    print("\nResults:")
    
    for i, result in enumerate(detailed_results):
        print(f"{i+1}. Input: {result['input']}")
        print(f"   Output: {result['output']}")
        print(f"   Time: {result['response_time_ms']:.2f} ms")
        print()

if __name__ == "__main__":
    main()