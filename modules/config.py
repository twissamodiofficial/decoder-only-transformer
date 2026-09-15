MHA_CHECKPOINT_DIR = "modules/checkpoints/mha"
MQA_CHECKPOINT_DIR = "modules/checkpoints/mqa"
GQA_CHECKPOINT_DIR = "modules/checkpoints/gqa"
TRAIN_DATA_PATH = "modules/data/data_loader/train_ids.npy"
VAL_DATA_PATH = "modules/data/data_loader/val_ids.npy"
TEST_DATA_PATH = "modules/data/data_loader/test_ids.npy"
TOKENIZER_PATH = "modules/data/tokenizer/wikitext_bpe_tokenizer.json"

VOCAB_SIZE = 8000
EMBED_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 6
MLP_RATIO = 4
NUM_KV_HEADS = 3

SEQ_LEN = 256
BATCH_SIZE = 64
MAX_SEQ_LEN = 512

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP_MAX_NORM = 1.0

NUM_EPOCHS = 20
SAVE_EVERY = 3

SEED = 42