# Gepeto-2

Implementacao educacional de um modelo de linguagem estilo **GPT-2** (Decoder-Only Transformer), construido do zero com PyTorch.

O objetivo nao e criar um modelo para producao, mas entender profundamente como funciona a arquitetura de um modelo autoregressivo baseado em Transformers — desde embeddings e atencao causal ate geracao de texto.

## Arquitetura

O modelo segue a arquitetura GPT-2 com algumas escolhas modernas:

- **Pre-LayerNorm** — normalizacao antes de attention e MLP, nao depois
- **Causal Self-Attention** — via `F.scaled_dot_product_attention` (FlashAttention na GPU)
- **GELU** como ativacao no MLP
- **Learned Positional Embeddings** — `nn.Embedding` ao inves de sinusoidal
- **AdamW** com gradient clipping e mixed precision (FP16)

Configuravel entre **10M e 120M parametros**, treinado localmente em uma **RTX 4060 (8GB VRAM)**.

## Estrutura do Projeto

```
gepeto/                     # Pacote Python (codigo do modelo)
  model.py                  # Classe GPT (nn.Module)
  dataset.py                # TextDataset (torch Dataset)
  tokenizer.py              # CharTokenizer (char-level)
  layers/
    attention.py             # CausalSelfAttention (multi-head)
    transformer_block.py     # TransformerBlock (Pre-LN + MLP)

train.py                    # Script de treinamento
generate.py                 # Script de geracao de texto

data/                       # Dados de treinamento e tokenizer
  corpus.txt                # Texto de treino
  tokenizer.json            # Vocabulario do tokenizer
```

## Uso

### Instalacao

```bash
pip install -r requirements.txt
```

Para suporte a CUDA (recomendado):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Treinamento

```bash
python train.py
```

O script detecta automaticamente se ha GPU disponivel. Checkpoints sao salvos em `checkpoints/`.

### Geracao de Texto

```bash
python generate.py
```

## Roadmap

- [x] Implementacao conceitual com NumPy
- [x] Migracao para PyTorch com suporte a GPU
- [ ] Evoluir tokenizer para BPE
- [ ] Coleta e preparacao de dataset educacional
- [ ] Treinamento de modelos de 10M+ parametros
- [ ] Escalonamento progressivo (gradient checkpointing)
- [ ] Estrategias de sampling (top-k, top-p, temperature)

## Referencias

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [nanoGPT (Karpathy)](https://github.com/karpathy/nanoGPT)
