import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gepeto import GPT, BPETokenizer, TextDataset, load_jsonl_corpus

# Presets de tamanho de modelo
PRESETS = {
    # Para smoke tests rapidos (segundos)
    "debug": dict(context_len=64,  embed_dim=64,  num_heads=4, num_layers=4, batch_size=64),
    # Para experimentos medios (minutos)
    "small": dict(context_len=128, embed_dim=128, num_heads=4, num_layers=6, batch_size=64),
    # Para treinamento serio (horas)
    "base":  dict(context_len=256, embed_dim=256, num_heads=8, num_layers=8, batch_size=32),
}


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
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
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Tokenizer e corpus
    tokenizer = BPETokenizer.load("data/bpe_tokenizer.json")
    print(f"Tokenizer: {tokenizer}")

    print("Encoding corpus...")
    encoded = load_jsonl_corpus(
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

    model = GPT(vocab_size, embed_dim, context_len, num_heads, num_layers).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    print(f"\nStart time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Acc':>8} | {'Time'}")
    print("-" * 60)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, device)

        now = datetime.now().strftime("%H:%M:%S")
        print(f"{epoch+1:5d} | {avg_train_loss:10.4f} | {val_loss:10.4f} | {val_acc:8.4f} | {now}")

    # Save checkpoint
    save_dir = os.path.join("checkpoints", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
            'context_len': context_len,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'tokenizer_type': 'bpe',
            'tokenizer_path': 'data/bpe_tokenizer.json',
        }
    }, os.path.join(save_dir, "checkpoint.pt"))
    print(f"\nModel saved to {save_dir}")


if __name__ == "__main__":
    main()
