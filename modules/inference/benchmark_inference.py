import time
import torch
import wandb

from modules.attention.CausalSelfAttentionWithKvCache import CausalSelfAttentionWithKvCache
from modules.attention.MultiQueryCausalAttention import MultiQueryCausalAttention
from modules.attention.GroupedQueryAttention import GroupedQueryAttention
from modules.transformer.gpt_clone import GPTClone
import modules.config as config


CHECKPOINTS = {
    "mha": (CausalSelfAttentionWithKvCache, f"{config.MHA_CHECKPOINT_DIR}/best.pt"),
    "gqa": (GroupedQueryAttention, f"{config.GQA_CHECKPOINT_DIR}/best.pt"),
    "mqa": (MultiQueryCausalAttention, f"{config.MQA_CHECKPOINT_DIR}/best.pt"),
}


def load_model(attention_module, checkpoint_path, device):
    model = GPTClone(
        config.VOCAB_SIZE,
        config.EMBED_DIM,
        config.NUM_HEADS,
        config.NUM_LAYERS,
        attention_module,
        config.MLP_RATIO,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def benchmark_generation(name, model, device, num_tokens=100, prompt_len=10):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    input_ids = torch.randint(0, config.VOCAB_SIZE, (1, prompt_len), device=device)
    kv_cache = None

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        # prefill: process the prompt once, building the initial cache
        logits, kv_cache = model(input_ids, kv_cache)
        next_token = logits[:, -1:, :].argmax(dim=-1)

        # decode: one new token at a time, reusing the cache
        for _ in range(num_tokens - 1):
            logits, kv_cache = model(next_token, kv_cache)
            next_token = logits[:, -1:, :].argmax(dim=-1)

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None

    print(
        f"{name}: generated {num_tokens} tokens in {elapsed:.3f}s "
        f"({num_tokens / elapsed:.1f} tok/s)"
        + (f", peak memory: {peak_mem_mb:.1f} MB" if peak_mem_mb is not None else "")
    )
    return elapsed, peak_mem_mb


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmark on {device}\n")

    wandb.init(project="gpt-clone-experiments", job_type="inference-benchmark")

    num_tokens = 100
    results = {}
    table = wandb.Table(columns=["attention", "elapsed_sec", "tokens_per_sec", "peak_memory_mb"])
    for name, (attention_module, checkpoint_path) in CHECKPOINTS.items():
        model = load_model(attention_module, checkpoint_path, device)
        elapsed, peak_mem = benchmark_generation(name, model, device, num_tokens=num_tokens)
        results[name] = (elapsed, peak_mem)

        tok_per_sec = num_tokens / elapsed
        wandb.log({
            f"benchmark/{name}/elapsed_sec": elapsed,
            f"benchmark/{name}/tokens_per_sec": tok_per_sec,
            f"benchmark/{name}/peak_memory_mb": peak_mem if peak_mem is not None else 0,
        })
        table.add_data(name, elapsed, tok_per_sec, peak_mem)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    wandb.log({"inference_benchmark_comparison": table})

    print("\n--- Summary ---")
    for name, (elapsed, peak_mem) in results.items():
        mem_str = f"{peak_mem:.1f} MB" if peak_mem is not None else "N/A"
        print(f"{name}: {elapsed:.3f}s, peak memory: {mem_str}")

    wandb.finish()