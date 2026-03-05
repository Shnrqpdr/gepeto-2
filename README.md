# Gepeto-2

Implementacao educacional de um modelo de linguagem estilo **GPT-2** (Decoder-Only Transformer), construido do zero com PyTorch.

## O que é esse projeto?

O objetivo nao e criar um modelo para producao, mas entender profundamente como funciona a arquitetura de um modelo autoregressivo baseado em Transformers — desde embeddings e atencao causal ate geracao de texto.

### Um completor de texto, não um assistente

O Gepeto-2 é um **base model** (modelo base): ele aprende a prever qual token vem a seguir numa sequência. Durante o treinamento, a única tarefa é:

> "dado que vi `[The, fundamental, theorem, of]`, qual é o próximo token?"

Fazendo isso milhões de vezes sobre um corpus de artigos de Matemática e Física, o modelo internaliza padrões estatísticos da linguagem. Ele aprende que depois de "The fundamental theorem of" provavelmente vem "calculus" — porque isso aparece frequentemente no corpus.

**O modelo não aprendeu nenhuma noção de pergunta, resposta, instrução ou tarefa.** Ele aprendeu distribuições de probabilidade sobre sequências de texto.

Por isso, ele vai brilhar em prompts com cara de texto contínuo:

```
In 1905, Albert Einstein published four papers that...
The Pythagorean theorem states that in a right triangle,...
```

E vai gerar resultados imprevisíveis em prompts conversacionais como "me explica o que é topologia" — porque esse estilo de linguagem não aparece em artigos da Wikipedia.

### Por que não é um ChatGPT?

Os LLMs modernos (ChatGPT, Claude, etc.) partem de um base model como este, mas passam por etapas adicionais:

1. **Supervised Fine-Tuning (SFT):** o modelo é treinado em milhares de pares `(instrução, resposta ideal)` escritos por humanos. Ele aprende o *formato* de conversa.
2. **RLHF (Reinforcement Learning from Human Feedback):** humanos avaliam as saídas. Um modelo de recompensa aprende o que humanos preferem. O modelo é ajustado para maximizar esse score — tornando respostas mais úteis, coerentes e seguras.

Sem essas etapas, o modelo é um base model: extraordinariamente capaz de modelar linguagem, mas sem o "modo assistente" ativado.

O Gepeto-2 vive intencionalmente nessa camada fundamental. O ponto é entender o que acontece antes de todo o fine-tuning — a parte que ninguém vê no produto final.

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
  dataset.py                # TextDataset + load_jsonl_corpus()
  tokenizer.py              # CharTokenizer (char-level, mantido para referencia)
  bpe_tokenizer.py          # BPETokenizer (byte-level BPE, vocab 8192)
  layers/
    attention.py             # CausalSelfAttention (multi-head)
    transformer_block.py     # TransformerBlock (Pre-LN + MLP)

train.py                    # Script de treinamento (presets: debug/small/base)
generate.py                 # Geracao interativa (REPL) ou modo prompt unico

scripts/
  train_tokenizer.py        # Treina o BPE tokenizer no corpus JSONL

data/
  bpe_tokenizer.json        # Tokenizer treinado
  scraping/
    wikipedia_scraper.py    # Scraper Wikipedia (API, saida JSONL)
    data/raw/
      wikipedia.jsonl       # ~1800 artigos de Matematica e Fisica

notes/                      # Registro de decisoes tecnicas
  001-bpe-tokenizer.md
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

### Treinamento do tokenizer

```bash
python scripts/train_tokenizer.py
```

### Treinamento do modelo

```bash
# Smoke test rapido (~30s)
python train.py --preset debug --max-tokens 50000 --epochs 3

# Experimento medio (minutos)
python train.py --preset small --max-tokens 500000

# Treinamento completo
python train.py --preset base
```

O script detecta automaticamente se ha GPU disponivel. Checkpoints sao salvos em `checkpoints/`.

### Geracao de texto

```bash
# Modo interativo (REPL)
python generate.py

# Prompt unico
python generate.py --prompt "The fundamental theorem of"
```

## Roadmap

- [x] Setup inicial PyTorch com suporte a GPU
- [x] Tokenizer BPE byte-level (vocab 8192, implementado do zero)
- [x] Coleta de dataset educacional (Wikipedia, ~27M tokens de Matematica e Fisica)
- [x] Geracao de texto interativa (modo REPL com temperature/top-k ajustaveis)
- [ ] Treinamento de modelos de 10M+ parametros
- [ ] Escalonamento progressivo (gradient checkpointing)
- [ ] Estrategias de sampling avancadas (top-p / nucleus sampling)
- [ ] Supervised Fine-Tuning para modo instrucao

## Referencias

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [nanoGPT (Karpathy)](https://github.com/karpathy/nanoGPT)
