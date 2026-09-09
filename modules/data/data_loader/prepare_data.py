from modules.data.tokenizer.tokenize_data import TokenizeData
import numpy as np

t = TokenizeData()
train_split = t.load_data('train')
t.load_tokenizer()
all_ids = []

for row in train_split:
    line = row['text'].strip()
    if line:
        ids = t.encode_tokens(line)
        all_ids.extend(ids)

all_ids = np.array(all_ids, dtype=np.uint16)
np.save('modules/data/data_loader/train_ids.npy', all_ids)