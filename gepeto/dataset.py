import json
import os

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, tokens, context_len):
        self.tokens = tokens
        self.context_len = context_len

    def __len__(self):
        return len(self.tokens) - self.context_len

    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.context_len]
        y = self.tokens[idx + 1:idx + self.context_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def load_jsonl_corpus(
    filepath: str,
    tokenizer,
    text_field: str = "text",
    max_tokens: int | None = None,
) -> list[int]:
    """Carrega JSONL, encoda cada documento e separa com <|endoftext|>.

    Se max_tokens for definido, para de ler assim que atingir o limite,
    sem precisar processar o corpus inteiro.
    """
    eot = tokenizer.encode("<|endoftext|>")
    all_tokens: list[int] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            tokens = tokenizer.encode(obj[text_field])
            all_tokens.extend(tokens)
            all_tokens.extend(eot)

            if max_tokens and len(all_tokens) >= max_tokens:
                break

    return all_tokens[:max_tokens] if max_tokens else all_tokens


def load_or_cache_corpus(
    filepath: str,
    tokenizer,
    cache_path: str = "data/corpus_tokens.pt",
    text_field: str = "text",
    max_tokens: int | None = None,
) -> list[int]:
    """Carrega tokens do cache se existir, senao encoda e salva.

    Para re-encodar (ex: apos atualizar o JSONL), basta deletar o cache
    ou rodar com --no-cache.
    """
    if not max_tokens and os.path.exists(cache_path):
        # Verifica se o cache é mais recente que o corpus
        corpus_mtime = os.path.getmtime(filepath)
        cache_mtime = os.path.getmtime(cache_path)
        if cache_mtime > corpus_mtime:
            print(f"Loading cached tokens from {cache_path}")
            tokens = torch.load(cache_path, weights_only=True).tolist()
            print(f"Cached tokens: {len(tokens):,}")
            return tokens
        print("Corpus modified since last cache, re-encoding...")

    tokens = load_jsonl_corpus(filepath, tokenizer, text_field, max_tokens)

    if not max_tokens:
        torch.save(torch.tensor(tokens, dtype=torch.int32), cache_path)
        print(f"Tokens cached to {cache_path}")

    return tokens
