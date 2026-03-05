"""
Treina o BPE tokenizer no corpus Wikipedia JSONL.

Uso:
  python scripts/train_tokenizer.py
  python scripts/train_tokenizer.py --vocab-size 4096
  python scripts/train_tokenizer.py --input data/raw/wikipedia.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add parent directory to path so we can import gepeto
sys.path.insert(0, str(Path(__file__).parent.parent))

from gepeto.bpe_tokenizer import BPETokenizer

DEFAULT_INPUT = Path("data/scraping/data/raw/wikipedia.jsonl")
DEFAULT_OUTPUT = Path("data/bpe_tokenizer.json")


def load_texts(filepath: Path) -> list[str]:
    texts = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina BPE tokenizer no corpus JSONL")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="JSONL de entrada")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Tokenizer de saida")
    parser.add_argument("--vocab-size", type=int, default=8192, help="Tamanho do vocabulario")
    args = parser.parse_args()

    print(f"Carregando textos de {args.input}...")
    texts = load_texts(args.input)
    total_chars = sum(len(t) for t in texts)
    print(f"  {len(texts)} documentos, {total_chars:,} caracteres")

    tokenizer = BPETokenizer(special_tokens=["<|endoftext|>"])

    print(f"\nTreinando BPE (vocab_size={args.vocab_size})...")
    t0 = time.time()
    tokenizer.fit(texts, vocab_size=args.vocab_size)
    elapsed = time.time() - t0
    print(f"  Tempo: {elapsed:.1f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.output))
    print(f"\nTokenizer salvo em {args.output}")

    # Stats
    print("\n--- Estatisticas ---")
    print(f"Vocab size: {tokenizer.vocab_size}")

    sample = texts[0][:2000]
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    ratio = len(sample.encode("utf-8")) / len(encoded)

    print(f"Amostra: {len(sample)} chars -> {len(encoded)} tokens (compressao: {ratio:.1f}x)")
    print(f"Roundtrip OK: {decoded == sample}")

    if decoded != sample:
        # Mostra onde diverge
        for i, (a, b) in enumerate(zip(decoded, sample)):
            if a != b:
                print(f"  Divergencia na posicao {i}: '{a}' vs '{b}'")
                print(f"  Contexto: ...{sample[max(0,i-20):i+20]}...")
                break

    # Test special token
    eot = tokenizer.encode("<|endoftext|>")
    print(f"<|endoftext|> -> {eot}")

    # Top merges
    print("\nTop 20 merges aprendidos:")
    for i, (a, b) in enumerate(tokenizer.merges[:20]):
        merged = tokenizer._vocab_table[256 + 1 + i]  # +1 for special token
        try:
            display = merged.decode("utf-8")
        except UnicodeDecodeError:
            display = repr(merged)
        print(f"  {i+1:3d}. ({a:>5}, {b:>5}) -> {repr(display)}")


if __name__ == "__main__":
    main()
