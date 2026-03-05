# 001 - De Char-level para BPE Tokenizer

**Data**: 2026-03-04

## Problema

O `CharTokenizer` proposto mapeava cada caractere a um token (vocab ~65). Com o
corpus Wikipedia (~27M caracteres, 614 chars unicos):

- Sequencias ficam enormes: "mathematics" = 11 tokens
- Janela de contexto de 64 tokens = apenas 64 caracteres de texto
- O modelo gasta capacidade aprendendo a juntar letras em palavras
- 296 chars raros (< 10 ocorrencias) viram `<UNK>` e se perdem

## Decisao: Byte Pair Encoding (BPE)

### Por que BPE?

BPE é o algoritmo usado pelo GPT-2 original. Aprende sub-palavras iterativamente
mergeando os pares de bytes mais frequentes. Com vocab=8192:

- **Compressao ~3-4x**: "mathematics" vira 1-2 tokens, nao 11
- **Mais contexto por janela**: 256 tokens BPE ~ 800-1000 caracteres
- **Tokens semanticos**: sub-palavras como "theorem", "equat" carregam significado
- **Educacional**: implementar do zero e um dos objetivos do projeto

### Por que byte-level?

O tokenizer opera sobre bytes (0-255), nao caracteres Unicode:

- **Zero UNK**: qualquer byte valido e representavel
- **Simples**: nao precisa enumerar o charset completo
- **Como GPT-2**: mesmo design do paper original

### Por que vocab_size = 8192?

- GPT-2 original usa 50257, mas para modelos de 768+ dim e datasets de GBs
- Com ~23MB de dados e modelos 10-120M params, embedding table de 50K seria desproporcional
- 8192 tokens: embedding table cabe tranquilo na VRAM, compressao suficiente
- Se o dataset crescer, pode-se retreinar com vocab maior

## Implementacao

### Arquivos criados/modificados

| Arquivo | Mudanca |
|---------|---------|
| `gepeto/bpe_tokenizer.py` | Classe BPETokenizer (byte-level, from scratch) |
| `scripts/train_tokenizer.py` | Script para treinar tokenizer no JSONL |
| `gepeto/dataset.py` | Adicionado `load_jsonl_corpus()` |
| `gepeto/__init__.py` | Exporta BPETokenizer e load_jsonl_corpus |
| `train.py` | Usa BPE + JSONL em vez de Char + corpus.txt |
| `generate.py` | Usa BPE |

### Algoritmo (resumo)

1. **Pre-tokenizacao**: regex estilo GPT-2 separa em palavras/pontuacao/whitespace
2. **Base vocab**: 256 bytes + special tokens (`<|endoftext|>`)
3. **Treinamento**: conta pares adjacentes, mergeia o mais frequente, repete
4. **Encode**: pre-tokeniza, converte para bytes, aplica merges em ordem
5. **Decode**: reconstroi bytes de cada token, decodifica UTF-8

### Hiperparametros do modelo atualizados

| Param | Antes (char) | Agora (BPE) | Razao |
|-------|-------------|-------------|-------|
| context_len | 64 | 256 | BPE tokens cobrem mais texto |
| embed_dim | 64 | 256 | Modelo maior para dataset maior |
| num_heads | 4 | 8 | Proporcional ao embed_dim |
| num_layers | 4 | 8 | Mais capacidade |
| batch_size | 64 | 32 | Compensa modelo maior na VRAM |
| train/val split | 80/20 | 90/10 | Mais dados de treino |

## Trade-offs

- **Encode e O(n*m)** onde n=seq_len, m=num_merges. Lento para textos longos.
  Implementacoes de producao usam priority queues, mas aqui priorizamos clareza.
- **CharTokenizer mantido** no pacote para comparacao e compatibilidade com
  checkpoints antigos.

## Como usar

```bash
# 1. Treinar o tokenizer
python scripts/train_tokenizer.py

# 2. Treinar o modelo
python train.py

# 3. Gerar texto
python generate.py
```

## Referencia

- Sennrich et al., 2016: "Neural Machine Translation of Rare Words with Subword Units"
- Radford et al., 2019: "Language Models are Unsupervised Multitask Learners" (GPT-2)
