from transformers import AutoTokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
print("Tokenizer loaded. You can now enter text to tokenize.")

while True:
    text = input("Enter text to tokenize (or 'exit' to quit): ")
    if text.lower() == 'exit':
        break
    tokens = tokenizer.tokenize(text)
    print("Tokens:", tokens)