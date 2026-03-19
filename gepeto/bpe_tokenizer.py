"""
Byte-level BPE tokenizer para o Gepeto-2.

Implementacao educacional do algoritmo Byte Pair Encoding, similar ao
tokenizer do GPT-2. Opera no nivel de bytes (nunca gera UNK) e aprende
merges a partir do corpus.

Referencia: https://arxiv.org/abs/1508.07909 (Sennrich et al., 2016)
"""

import json
import re
from collections import Counter
from typing import Optional

try:
    from .cbpe import apply_merges as _c_apply_merges, apply_merges_batch as _c_apply_merges_batch
    _HAS_C = True
except Exception:
    _HAS_C = False


# Regex de pre-tokenizacao estilo GPT-2.
# Separa em: contracoes, palavras (com espaco opcional antes),
# numeros, pontuacao, whitespace.
GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d"""
    r"""| ?\w+"""
    r"""| ?\d+"""
    r"""| ?[^\s\w\d]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+""",
    re.UNICODE,
)


class BPETokenizer:

    def __init__(self, special_tokens: Optional[list[str]] = None):
        # IDs 0-255: bytes individuais
        self.merges: list[tuple[int, int]] = []
        self.special_tokens: dict[str, int] = {}
        self._vocab_table: dict[int, bytes] = {}
        self.vocab_size = 256
        self._is_fitted = False

        if special_tokens:
            self.add_special_tokens(special_tokens)

    # ------------------------------------------------------------------ #
    # Special tokens
    # ------------------------------------------------------------------ #

    def add_special_tokens(self, tokens: list[str]) -> None:
        for token in tokens:
            if token not in self.special_tokens:
                token_id = 256 + len(self.special_tokens)
                self.special_tokens[token] = token_id
                self._vocab_table[token_id] = token.encode("utf-8")
                self.vocab_size = max(self.vocab_size, token_id + 1)

    # ------------------------------------------------------------------ #
    # Treinamento (fit)
    # ------------------------------------------------------------------ #

    def fit(self, texts: list[str], vocab_size: int = 8192) -> None:
        """Aprende merges BPE a partir de uma lista de textos."""
        num_special = len(self.special_tokens)
        num_merges = vocab_size - 256 - num_special

        if num_merges <= 0:
            raise ValueError(
                f"vocab_size ({vocab_size}) deve ser > 256 + {num_special} special tokens"
            )

        # Pre-tokeniza e conta frequencia de cada chunk
        chunk_freq: dict[tuple[int, ...], int] = Counter()
        for i, text in enumerate(texts):
            for word in GPT2_PAT.findall(text):
                chunk = tuple(word.encode("utf-8"))
                chunk_freq[chunk] += 1
            if (i + 1) % 500 == 0:
                print(f"  Pre-tokenizacao: {i + 1}/{len(texts)} textos")

        print(f"  {len(chunk_freq):,} chunks unicos encontrados")

        # Converte para listas mutaveis
        chunks = {i: list(chunk) for i, chunk in enumerate(chunk_freq.keys())}
        freqs = list(chunk_freq.values())

        self.merges = []
        next_id = 256 + num_special

        for merge_idx in range(num_merges):
            # Conta pares adjacentes
            pair_counts: dict[tuple[int, int], int] = {}
            for idx, chunk in chunks.items():
                freq = freqs[idx]
                for j in range(len(chunk) - 1):
                    pair = (chunk[j], chunk[j + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.__getitem__)
            new_id = next_id + merge_idx
            self.merges.append(best_pair)

            # Aplica merge em todos os chunks
            for idx, chunk in chunks.items():
                new_chunk = []
                i = 0
                while i < len(chunk):
                    if (
                        i < len(chunk) - 1
                        and chunk[i] == best_pair[0]
                        and chunk[i + 1] == best_pair[1]
                    ):
                        new_chunk.append(new_id)
                        i += 2
                    else:
                        new_chunk.append(chunk[i])
                        i += 1
                chunks[idx] = new_chunk

            if (merge_idx + 1) % 500 == 0:
                print(f"  Merge {merge_idx + 1}/{num_merges} (pair: {best_pair})")

        self.vocab_size = 256 + num_special + len(self.merges)
        self._build_vocab_table()
        self._build_c_merge_arrays()
        self._is_fitted = True
        print(f"  Treinamento completo. vocab_size = {self.vocab_size}")

    # ------------------------------------------------------------------ #
    # Encode / Decode
    # ------------------------------------------------------------------ #

    def encode(self, text: str) -> list[int]:
        if not self._is_fitted:
            raise ValueError("Tokenizer must be fitted before encoding")

        # Separa special tokens do texto
        segments = self._split_special_tokens(text)
        result: list[int] = []

        for segment, is_special in segments:
            if is_special:
                result.append(self.special_tokens[segment])
            elif _HAS_C:
                # Coleta todos os chunks do segmento e processa em batch via C
                chunks = [list(w.encode("utf-8")) for w in GPT2_PAT.findall(segment)]
                if chunks:
                    merged = _c_apply_merges_batch(chunks, self._c_merges_a, self._c_merges_b, self._c_base_id)
                    for m in merged:
                        result.extend(m)
            else:
                for word in GPT2_PAT.findall(segment):
                    tokens = list(word.encode("utf-8"))
                    tokens = self._apply_merges(tokens)
                    result.extend(tokens)

        return result

    def decode(self, tokens: list[int]) -> str:
        raw_bytes = b""
        for token_id in tokens:
            if token_id in self._vocab_table:
                raw_bytes += self._vocab_table[token_id]
            elif 0 <= token_id < 256:
                raw_bytes += bytes([token_id])
            else:
                raw_bytes += b"?"
        return raw_bytes.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    # Save / Load
    # ------------------------------------------------------------------ #

    def save(self, filepath: str) -> None:
        data = {
            "version": "bpe-v1",
            "vocab_size": self.vocab_size,
            "merges": [list(pair) for pair in self.merges],
            "special_tokens": self.special_tokens,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(filepath: str) -> "BPETokenizer":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = BPETokenizer()
        tok.merges = [tuple(pair) for pair in data["merges"]]
        tok.special_tokens = data.get("special_tokens", {})
        tok.vocab_size = data["vocab_size"]
        tok._build_vocab_table()
        tok._build_c_merge_arrays()
        tok._is_fitted = True
        return tok

    # ------------------------------------------------------------------ #
    # Helpers internos
    # ------------------------------------------------------------------ #

    def _build_c_merge_arrays(self) -> None:
        """Pre-computa arrays de merges para o backend C."""
        self._c_merges_a = [a for a, b in self.merges]
        self._c_merges_b = [b for a, b in self.merges]
        self._c_base_id = 256 + len(self.special_tokens)

    def _build_vocab_table(self) -> None:
        """Constroi tabela token_id -> bytes para decode rapido."""
        table: dict[int, bytes] = {}

        # Bytes base
        for i in range(256):
            table[i] = bytes([i])

        # Special tokens
        for token_str, token_id in self.special_tokens.items():
            table[token_id] = token_str.encode("utf-8")

        # Merges
        base_id = 256 + len(self.special_tokens)
        for i, (a, b) in enumerate(self.merges):
            token_id = base_id + i
            table[token_id] = table[a] + table[b]

        self._vocab_table = table

    def _apply_merges(self, tokens: list[int]) -> list[int]:
        """Aplica todos os merges aprendidos a uma lista de tokens."""
        base_id = 256 + len(self.special_tokens)

        for i, (a, b) in enumerate(self.merges):
            new_id = base_id + i
            j = 0
            new_tokens = []
            while j < len(tokens):
                if (
                    j < len(tokens) - 1
                    and tokens[j] == a
                    and tokens[j + 1] == b
                ):
                    new_tokens.append(new_id)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens

        return tokens

    def _split_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        """Divide texto em segmentos, separando special tokens."""
        if not self.special_tokens:
            return [(text, False)]

        # Escapa os tokens para regex e ordena por tamanho (maior primeiro)
        sorted_tokens = sorted(self.special_tokens.keys(), key=len, reverse=True)
        pattern = "|".join(re.escape(t) for t in sorted_tokens)

        segments: list[tuple[str, bool]] = []
        last_end = 0

        for match in re.finditer(pattern, text):
            if match.start() > last_end:
                segments.append((text[last_end:match.start()], False))
            segments.append((match.group(), True))
            last_end = match.end()

        if last_end < len(text):
            segments.append((text[last_end:], False))

        return segments

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={self.vocab_size}, fitted={self._is_fitted})"
