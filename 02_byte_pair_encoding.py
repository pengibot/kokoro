import json
import os
import pickle
import sys
sys.path.append('..')

from minbpe import BasicTokenizer

def train_tokenizer():
    with open('output/combined_text.txt', 'r', encoding="utf-8") as f:
        text_sequence = f.readline()

    tokenizer = BasicTokenizer()
    tokenizer.train(text_sequence, vocab_size=1024)

    vocab = tokenizer.vocab
    with open('output/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)


def load_tokenizer():
    with open('output/tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

# train_tokenizer()
tokenizer = load_tokenizer()
vocab = tokenizer.vocab
# for key in tokenizer.vocab.keys():
#     try:
#         print(tokenizer.vocab[key].decode('utf-8'))
#     except UnicodeDecodeError:
#         continue

# encoded = tokenizer.encode('なん')
# print(encoded)
# decoded = tokenizer.decode([622])
# print(decoded)

max_vocab_id = list(vocab.keys())[-1]
tokenizer.special_tokens = {
    "<|startoftext|>": max_vocab_id + 1,
    "<|separator|>": max_vocab_id + 2,
    "<|endoftext|>": max_vocab_id + 3,
    "<|unk|>": max_vocab_id + 4
}

with open("output/combined_text.txt", "r", encoding="utf-8") as f:
    text_sequence = f.readline()
length = len(tokenizer.encode(text_sequence))
print(length)

os.makedirs("output/tokenizer", exist_ok=True)
tokenizer.save(file_prefix="output/tokenizer/my_tokenizer")