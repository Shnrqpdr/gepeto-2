import argparse
import glob

import torch

from gepeto import GPT, BPETokenizer


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint['config']
    model_keys = {'vocab_size', 'embed_dim', 'context_len', 'num_heads', 'num_layers'}
    model_config = {k: v for k, v in config.items() if k in model_keys}
    model = GPT(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def latest_checkpoint() -> str | None:
    """Retorna o checkpoint mais recente em checkpoints/."""
    dirs = sorted(glob.glob("checkpoints/*/checkpoint.pt"))
    return dirs[-1] if dirs else None


def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=40):
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

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

        token_str = tokenizer.decode([next_token.item()])
        print(token_str, end="", flush=True)

    print()


def interactive_loop(model, tokenizer, max_new_tokens, temperature, top_k):
    print("\n=== Gepeto-2 | modo interativo ===")
    print("Digite um prompt e pressione Enter para continuar o texto.")
    print("Comandos: :sair  :temperatura <valor>  :tokens <n>  :topk <n>\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAteh mais!")
            break

        if not prompt:
            continue

        # Comandos internos
        if prompt == ":sair":
            print("Ate mais!")
            break
        if prompt.startswith(":temperatura "):
            temperature = float(prompt.split()[1])
            print(f"  temperatura = {temperature}")
            continue
        if prompt.startswith(":tokens "):
            max_new_tokens = int(prompt.split()[1])
            print(f"  max_new_tokens = {max_new_tokens}")
            continue
        if prompt.startswith(":topk "):
            top_k = int(prompt.split()[1])
            print(f"  top_k = {top_k}")
            continue

        print()
        generate_text(model, tokenizer, prompt, max_new_tokens, temperature, top_k)
        print()


def main():
    parser = argparse.ArgumentParser(description="Geracao de texto interativa com Gepeto-2")
    parser.add_argument("--checkpoint", default=None, help="Caminho do checkpoint (padrao: mais recente)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--prompt", default=None, help="Prompt unico (sem modo interativo)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    checkpoint = args.checkpoint or latest_checkpoint()
    if not checkpoint:
        print("Nenhum checkpoint encontrado em checkpoints/. Treine o modelo primeiro.")
        return
    print(f"Checkpoint: {checkpoint}")

    tokenizer = BPETokenizer.load("data/bpe_tokenizer.json")
    model = load_model(checkpoint, device)
    print(f"Modelo carregado ({model.count_parameters():,} params)")

    if args.prompt:
        # Modo nao interativo: gera e sai
        print(f"\n{args.prompt}", end="")
        generate_text(model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_k)
    else:
        interactive_loop(model, tokenizer, args.max_tokens, args.temperature, args.top_k)


if __name__ == "__main__":
    main()
