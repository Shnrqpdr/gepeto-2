# 002 - Encoding BPE em C via ctypes

**Data**: 2026-03-18

## Problema

O encoding do corpus inteiro (~27M chars, ~1800 artigos Wikipedia) com o
BPE tokenizer em Python puro levava mais de 1 hora. O gargalo é o
`_apply_merges`: para cada palavra, aplica ~7900 merges em sequência,
cada um varrendo o array de tokens. Em Python, cada iteração carrega o
overhead do interpretador (indireção de ponteiros, type checking dinâmico,
GC, objetos int no heap).

Com planos de treinar em múltiplos corpus diferentes, esperar 1h+ por
encoding não era viável mesmo com cache em disco.

## Decisão: reescrever o core em C

O `_apply_merges` é um loop duplo sobre arrays de inteiros. Não depende
de nenhuma feature do Python. Candidato perfeito para C.

### O que foi feito

| Arquivo | O que faz |
|---------|-----------|
| `gepeto/cbpe/bpe_merge.c` | `apply_merges` (1 chunk) e `apply_merges_batch` (N chunks) |
| `gepeto/cbpe/__init__.py` | Wrapper ctypes, compila .so automaticamente |
| `gepeto/bpe_tokenizer.py` | `encode()` usa backend C quando disponível, fallback Python |

O algoritmo em C é idêntico ao Python. Mesmos loops, mesma lógica. A
diferença é toda na execução: array contíguo de `int`, acesso direto
à memória, sem overhead de interpretador.

### Compilação automática

O `.c` compila na primeira importação se a `.so` não existir ou se o
fonte for mais recente. Se o gcc não estiver disponível, o tokenizer
continua funcionando com a implementação Python pura.

## Benchmark

100 artigos (1.5M chars):

| Backend | Tempo   | Tokens    | Speedup |
|---------|---------|-----------|---------|
| Python  | 326.2s  | 344,620   | 1x      |
| C       | 3.6s    | 344,620   | 89.5x   |

Corpus inteiro (~27M chars): ~50 segundos em C (estimado 1h40min em Python, ~116x).

Saída idêntica token por token.

## Por que não OpenMP

Os chunks são independentes, então paralelizar com OpenMP seria direto.
Mas com ~100x de speedup single-threaded, o corpus inteiro já encoda em
menos de 1 minuto. OpenMP fica como opção se o corpus crescer para GBs.

## Complemento: cache de tokens

Além do backend C, implementamos cache em disco (`data/corpus_tokens.pt`).
Na primeira execução sem `--max-tokens`, os tokens são salvos como tensor.
Nas execuções seguintes, carrega direto. Se o JSONL for modificado, o cache
detecta pelo mtime e re-encoda automaticamente.

As duas soluções se complementam: C para o primeiro encoding de qualquer
corpus, cache para pular o encoding em execuções repetidas.
