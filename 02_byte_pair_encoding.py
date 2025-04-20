import os
import pickle
import sys
sys.path.append('..')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from minbpe import BasicTokenizer

def train_tokenizer():
    with open('output/combined_text.txt', 'r', encoding="utf-8") as f:
        text_sequence = f.readline()

    tokenizer = BasicTokenizer()
    tokenizer.train(text_sequence, vocab_size=10000)

    with open('output/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)


def load_tokenizer():
    with open('output/tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

# Testing methods work
train_tokenizer()
tokenizer = load_tokenizer()

# Printing out some of the tokens to check it worked
# for key in tokenizer.vocab.keys():
#     try:
#         print(tokenizer.vocab[key].decode('utf-8'))
#     except UnicodeDecodeError:
#         continue

# Validating the encoding and decoding works
# encoded = tokenizer.encode('なん')
# print(encoded)
# decoded = tokenizer.decode([622])
# print(decoded)

vocab = tokenizer.vocab
max_vocab_id = list(vocab.keys())[-1]
tokenizer.special_tokens = {
    "<|startoftext|>": max_vocab_id + 1,
    "<|separator|>": max_vocab_id + 2,
    "<|endoftext|>": max_vocab_id + 3,
    "<|unk|>": max_vocab_id + 4
}

# Check out how many tokens it generates (853684)
# with open("output/combined_text.txt", "r", encoding="utf-8") as f:
#     text_sequence = f.readline()
# length = len(tokenizer.encode(text_sequence))
# print(length)

os.makedirs("output/tokenizer", exist_ok=True)
tokenizer.save(file_prefix="output/tokenizer/my_tokenizer")