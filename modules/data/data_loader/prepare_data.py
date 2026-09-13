from modules.data.tokenizer.tokenize_data import TokenizeData
import numpy as np
import os
import modules.config as config


def tokenize_split(tokenizer, split_name):
    split_data = tokenizer.load_data(split_name)
    all_ids = []
    for row in split_data:
        line = row['text'].strip()
        if line:
            ids = tokenizer.encode_tokens(line)
            all_ids.extend(ids)
    return np.array(all_ids, dtype=np.uint16)


def prepare_data():
    if not os.path.exists(config.TRAIN_DATA_PATH) or not os.path.exists(config.VAL_DATA_PATH):
        data_tokenizer = TokenizeData()

        if os.path.exists(config.TOKENIZER_PATH):
            data_tokenizer.load_tokenizer()
        else:
            data_tokenizer.train_wikitext_tokenizer()

        if not os.path.exists(config.TRAIN_DATA_PATH):
            train_ids = tokenize_split(data_tokenizer, 'train')
            np.save(config.TRAIN_DATA_PATH, train_ids)
            print(f"Saved {len(train_ids)} train tokens to {config.TRAIN_DATA_PATH}")

        if not os.path.exists(config.VAL_DATA_PATH):
            val_ids = tokenize_split(data_tokenizer, 'validation')
            np.save(config.VAL_DATA_PATH, val_ids)
            print(f"Saved {len(val_ids)} val tokens to {config.VAL_DATA_PATH}")
    else:
        print("Tokenized data already exists, skipping tokenization.")

if __name__ == "__main__":
    prepare_data()