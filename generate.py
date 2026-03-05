import torch
from gepeto import GPT, BPETokenizer


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint['config']
    # Remove campos extras do config que nao sao parametros do modelo
    model_keys = {'vocab_size', 'embed_dim', 'context_len', 'num_heads', 'num_layers'}
    model_config = {k: v for k, v in config.items() if k in model_keys}
    model = GPT(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=40):
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    print(prompt, end="", flush=True)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            idx_cond = idx[:, -model.context_len:]
            logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        char = tokenizer.decode([next_token.item()])
        print(char, end="", flush=True)

    print()
    return tokenizer.decode(idx[0].tolist())


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = BPETokenizer.load("data/bpe_tokenizer.json")
    model = load_model("checkpoints/<TIMESTAMP>/checkpoint.pt", device)

    prompt = "The fundamental theorem of"
    generate_text(model, tokenizer, prompt, max_new_tokens=200)
