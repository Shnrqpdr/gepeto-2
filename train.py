import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from gepeto import GPT, TextDataset, CharTokenizer
from datetime import datetime
import os


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


def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    device = get_device()
    print(f"Using device: {device}")

    raw_text = load_text("data/corpus.txt")
    tokenizer = CharTokenizer.load("data/tokenizer.json")
    encoded = tokenizer.encode(raw_text)

    # Train/val split
    split_idx = int(len(encoded) * 0.8)
    train_tokens = encoded[:split_idx]
    val_tokens = encoded[split_idx:]

    # Hyperparameters
    context_len = 64
    embed_dim = 64
    num_heads = 4
    num_layers = 4
    batch_size = 64
    learning_rate = 3e-4
    epochs = 10
    vocab_size = tokenizer.vocab_size

    train_dataset = TextDataset(train_tokens, context_len)
    val_dataset = TextDataset(val_tokens, context_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = GPT(vocab_size, embed_dim, context_len, num_heads, num_layers).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    print(f"\nStart time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Acc':>8} | {'Time'}")
    print("-" * 60)

    for epoch in range(epochs):
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
        }
    }, os.path.join(save_dir, "checkpoint.pt"))
    print(f"\nModel saved to {save_dir}")


if __name__ == "__main__":
    main()
