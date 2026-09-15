import argparse
import torch
from torch.utils.data import DataLoader
import wandb

from modules.attention.CausalSelfAttentionWithKvCache import CausalSelfAttentionWithKvCache
from modules.attention.MultiQueryCausalAttention import MultiQueryCausalAttention
from modules.attention.GroupedQueryAttention import GroupedQueryAttention
from modules.training.train_transformer import TransformerTrainer
from modules.data.data_loader.token_dataset import TokenDataset
import modules.config as config
from modules.data.data_loader.prepare_data import prepare_data


ATTENTION_MODULES = {
    "mha": CausalSelfAttentionWithKvCache,
    "mqa": MultiQueryCausalAttention,
    "gqa": GroupedQueryAttention,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attention",
        type=str,
        choices=ATTENTION_MODULES.keys(),
        default="mha",
        help="Which attention module to use: mha, gqa or mqa",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(config.SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    prepare_data()

    attention_module = ATTENTION_MODULES[args.attention]
    mlp_ratio = config.MLP_RATIO
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = TokenDataset(config.TRAIN_DATA_PATH, config.SEQ_LEN)
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    val_dataset = TokenDataset(config.VAL_DATA_PATH, config.SEQ_LEN)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    wandb.init(
        project="gpt-clone-experiments",
        config={
            "attention": args.attention,
            "vocab_size": config.VOCAB_SIZE,
            "embed_dim": config.EMBED_DIM,
            "num_heads": config.NUM_HEADS,
            "num_layers": config.NUM_LAYERS,
            "mlp_ratio": mlp_ratio,
            "seq_len": config.SEQ_LEN,
            "batch_size": config.BATCH_SIZE,
            "lr": config.LEARNING_RATE,
            "weight_decay": config.WEIGHT_DECAY,
            "epochs": config.NUM_EPOCHS,
        },
    )

    trainer = TransformerTrainer(
        config.VOCAB_SIZE,
        config.EMBED_DIM,
        config.NUM_HEADS,
        config.NUM_LAYERS,
        attention_module,
        mlp_ratio,
        device,
    )
    print(f"TransformerTrainer initialized with {args.attention} attention on {device}.")

    CHECKPOINT_DIRS = {
        "mha": config.MHA_CHECKPOINT_DIR,
        "mqa": config.MQA_CHECKPOINT_DIR,
        "gqa": config.GQA_CHECKPOINT_DIR,
    }

    trainer.train(
        train_dataloader,
        epochs=config.NUM_EPOCHS,
        save_every=config.SAVE_EVERY,
        val_dataloader=val_dataloader,
        checkpoint_dir=CHECKPOINT_DIRS[args.attention],
    )