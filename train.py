import argparse
import csv
import math
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gepeto import GPT, BPETokenizer, TextDataset, load_or_cache_corpus

# Presets de tamanho de modelo
PRESETS = {
    # Para smoke tests rapidos (segundos)
    "debug":  dict(context_len=64,  embed_dim=64,  num_heads=4,  num_layers=4,  batch_size=64),
    # Para experimentos medios (minutos)
    "small":  dict(context_len=128, embed_dim=128, num_heads=4,  num_layers=6,  batch_size=64),
    # Para treinamento serio (horas)
    "base":   dict(context_len=256, embed_dim=256, num_heads=8,  num_layers=8,  batch_size=32),
    # ~25M params — RTX 5070 12GB com grad accum
    "medium": dict(context_len=512, embed_dim=512, num_heads=8,  num_layers=12, batch_size=16),
    # ~85M params — RTX 5070 12GB com grad accum e batch menor
    "large":  dict(context_len=512, embed_dim=768, num_heads=12, num_layers=12, batch_size=8),
}


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_amp_dtype(device):
    """Detecta o melhor dtype para AMP baseado no hardware.
    BF16 em GPUs com suporte (Ampere+), FP16 nas demais.
    """
    if device.type != 'cuda':
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@torch.no_grad()
def evaluate(model, dataloader, device, amp_dtype):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_tokens += y.numel()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / total_tokens
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Treina o Gepeto-2")
    parser.add_argument(
        "--preset", choices=PRESETS.keys(), default="base",
        help="Preset de tamanho do modelo (default: base)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Limita o corpus a N tokens (util para testes rapidos)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Numero de epochs (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=1,
        help="Gradient accumulation steps (default: 1, effective batch = batch_size * grad_accum)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Caminho para checkpoint para retomar treino",
    )
    parser.add_argument(
        "--grad-checkpoint", action="store_true",
        help="Ativa gradient checkpointing (troca tempo por VRAM)",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Tokenizer e corpus
    tokenizer = BPETokenizer.load("data/bpe_tokenizer.json")
    print(f"Tokenizer: {tokenizer}")

    print("Loading corpus...")
    encoded = load_or_cache_corpus(
        "data/scraping/data/raw/wikipedia.jsonl", tokenizer, max_tokens=args.max_tokens
    )
    print(f"Total tokens: {len(encoded):,}")

    # Train/val split
    split_idx = int(len(encoded) * 0.9)
    train_tokens = encoded[:split_idx]
    val_tokens = encoded[split_idx:]

    # Hiperparametros do preset escolhido
    hp = PRESETS[args.preset]
    context_len = hp["context_len"]
    embed_dim   = hp["embed_dim"]
    num_heads   = hp["num_heads"]
    num_layers  = hp["num_layers"]
    batch_size  = hp["batch_size"]
    vocab_size  = tokenizer.vocab_size

    print(f"Preset: {args.preset} | ctx={context_len} dim={embed_dim} heads={num_heads} layers={num_layers}")

    train_dataset = TextDataset(train_tokens, context_len)
    val_dataset   = TextDataset(val_tokens, context_len)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader    = DataLoader(val_dataset, batch_size=batch_size)

    amp_dtype = get_amp_dtype(device)
    use_scaler = (amp_dtype == torch.float16)
    print(f"AMP dtype: {amp_dtype} | GradScaler: {'on' if use_scaler else 'off'}")

    model = GPT(vocab_size, embed_dim, context_len, num_heads, num_layers,
                gradient_checkpointing=args.grad_checkpoint).to(device)
    if args.grad_checkpoint:
        print("Gradient checkpointing: on")
    print(f"Model parameters: {model.count_parameters():,}")

    # Weight decay seletivo: decay em pesos de Linear, sem decay em biases e LayerNorm
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or 'ln' in name or 'bias' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    # Cosine LR scheduler com linear warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(total_steps // 10, 2000)
    min_lr = args.lr * 0.1

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr / args.lr, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Diretorio do run (criado antes para salvar logs durante o treino)
    save_dir = os.path.join("checkpoints", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(save_dir, exist_ok=True)

    # Checkpoint resumption
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(resume_ckpt['model_state_dict'])
        if 'optimizer_state_dict' in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in resume_ckpt:
            scaler.load_state_dict(resume_ckpt['scaler_state_dict'])
        if 'epoch' in resume_ckpt:
            start_epoch = resume_ckpt['epoch']
            print(f"Resuming from epoch {start_epoch + 1}")

    grad_accum_steps = args.grad_accum
    effective_batch = batch_size * grad_accum_steps

    log_path = os.path.join(save_dir, "metrics.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "lr", "grad_norm"])

    print(f"\nStart time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Total steps: {total_steps:,} | Warmup: {warmup_steps:,} | Grad accum: {grad_accum_steps} | Effective batch: {effective_batch}")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Acc':>8} | {'LR':>10} | {'Grad Norm':>10} | {'Time'}")
    print("-" * 85)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        epoch_grad_norm = 0
        optimizer_steps = 0

        optimizer.zero_grad()
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                epoch_grad_norm += grad_norm
                optimizer_steps += 1
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_loader)
        avg_grad_norm = epoch_grad_norm / max(1, optimizer_steps)
        val_loss, val_acc = evaluate(model, val_loader, device, amp_dtype)
        current_lr = optimizer.param_groups[0]['lr']

        log_writer.writerow([epoch + 1, f"{avg_train_loss:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}", f"{current_lr:.2e}", f"{avg_grad_norm:.4f}"])
        log_file.flush()

        now = datetime.now().strftime("%H:%M:%S")
        print(f"{epoch+1:5d} | {avg_train_loss:10.4f} | {val_loss:10.4f} | {val_acc:8.4f} | {current_lr:10.2e} | {avg_grad_norm:10.4f} | {now}")

    log_file.close()

    # Save checkpoint (com estado completo para resumption)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'epoch': args.epochs,
        'config': {
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
            'context_len': context_len,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'dropout': 0.1,
            'tokenizer_type': 'bpe',
            'tokenizer_path': 'data/bpe_tokenizer.json',
        }
    }, os.path.join(save_dir, "checkpoint.pt"))
    print(f"\nModel saved to {save_dir}")


if __name__ == "__main__":
    main()
