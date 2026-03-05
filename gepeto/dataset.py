import json

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


def load_jsonl_corpus(filepath: str, tokenizer, text_field: str = "text") -> list[int]:
    """Carrega JSONL, encoda cada documento e separa com <|endoftext|>."""
    eot = tokenizer.encode("<|endoftext|>")
    all_tokens: list[int] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            tokens = tokenizer.encode(obj[text_field])
            all_tokens.extend(tokens)
            all_tokens.extend(eot)

    return all_tokens
