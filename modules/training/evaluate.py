import torch
import wandb
from torch.utils.data import DataLoader

from modules.attention.CausalSelfAttentionWithKvCache import CausalSelfAttentionWithKvCache
from modules.attention.MultiQueryCausalAttention import MultiQueryCausalAttention
from modules.attention.GroupedQueryAttention import GroupedQueryAttention
from modules.training.train_transformer import TransformerTrainer
from modules.data.data_loader.token_dataset import TokenDataset
import modules.config as config


CHECKPOINTS = {
    "mha": (CausalSelfAttentionWithKvCache, f"{config.MHA_CHECKPOINT_DIR}/best.pt"),
    "gqa": (GroupedQueryAttention, f"{config.GQA_CHECKPOINT_DIR}/best.pt"),
    "mqa": (MultiQueryCausalAttention, f"{config.MQA_CHECKPOINT_DIR}/best.pt"),
}


def evaluate_checkpoint(name, attention_module, checkpoint_path, test_dataloader, device):
    trainer = TransformerTrainer(
        config.VOCAB_SIZE,
        config.EMBED_DIM,
        config.NUM_HEADS,
        config.NUM_LAYERS,
        attention_module,
        config.MLP_RATIO,
        device,
    )
    trainer.load_checkpoint(checkpoint_path)
    test_loss = trainer.validate(test_dataloader)
    print(f"{name}: test_loss = {test_loss:.4f}  (from {checkpoint_path})")
    return test_loss


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project="gpt-clone-experiments", job_type="test-evaluation")

    test_dataset = TokenDataset(config.TEST_DATA_PATH, config.SEQ_LEN)
    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    results = {}
    table = wandb.Table(columns=["attention", "test_loss"])
    for name, (attention_module, checkpoint_path) in CHECKPOINTS.items():
        loss = evaluate_checkpoint(name, attention_module, checkpoint_path, test_dataloader, device)
        results[name] = loss
        wandb.log({f"test_loss/{name}": loss})
        table.add_data(name, loss)

    wandb.log({"test_loss_comparison": table})

    print("\n--- Summary ---")
    for name, loss in results.items():
        print(f"{name}: {loss:.4f}")

    wandb.finish()