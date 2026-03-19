# 003 - Otimizações de VRAM para escalar o modelo

**Data**: 2026-03-18

## Contexto

RTX 5070 com 12GB de VRAM. Queremos treinar modelos de até ~85M parâmetros
(preset `large`: dim=768, 12 layers, ctx=512) e eventualmente maiores. A
VRAM é o fator limitante.

## O que consome VRAM durante o treino

Quando você treina um transformer, a VRAM é dividida entre:

1. **Parâmetros do modelo** - os pesos em si (~340MB para 85M params em FP16)
2. **Estado do otimizador** - AdamW guarda momentum e variância por parâmetro,
   ou seja, 2x o tamanho do modelo (~680MB para 85M params)
3. **Ativações** - saídas intermediárias de cada layer, guardadas para o backward.
   Cresce com: batch_size × context_len × embed_dim × num_layers. Este é o
   maior consumidor e o que mais varia.
4. **Gradientes** - mesmos tamanho dos parâmetros (~340MB)

Para 85M params com batch=8, ctx=512, as ativações facilmente dominam e
podem consumir 4-6GB sozinhas.

## Gradient checkpointing

### O tradeoff

Em vez de guardar as ativações de todos os layers para o backward pass,
guarda apenas a entrada de cada bloco transformer. No backward, recomputa
as ativações sob demanda. Troca ~30% mais tempo de treino por uma redução
grande no uso de VRAM de ativações.

### Implementação

O modelo aceita `gradient_checkpointing=True` no construtor. No forward,
cada bloco usa `torch.utils.checkpoint.checkpoint()` quando em modo
training. Em eval (generate), desliga automaticamente.

`nn.Sequential` foi trocado por `nn.ModuleList` para iterar bloco a bloco.

Flag `--grad-checkpoint` no train.py.

## BF16 vs FP16 (detecção automática por hardware)

### O problema do FP16

FP16 tem range dinâmico limitado (max ~65504). Gradientes podem dar overflow,
por isso precisa de GradScaler. O scaler consome memória e adiciona
complexidade.

### BF16

BF16 (bfloat16) tem o mesmo range do FP32 (8 bits de expoente) com precisão
reduzida (7 bits de mantissa vs 10 do FP16). Na prática: não precisa de
GradScaler, e é mais estável para treino.

### Compatibilidade

Nem toda GPU suporta BF16. A detecção é automática:

- `torch.cuda.is_bf16_supported()` retorna True → usa BF16, desliga GradScaler
- Senão → usa FP16 com GradScaler (comportamento anterior)
- CPU → float32

Isso mantém compatibilidade com a RTX 4060 (FP16) e aproveita BF16 na
RTX 5070 (Blackwell).

## Outras opções não implementadas (para referência futura)

- **8-bit Adam** (bitsandbytes): reduz estado do otimizador de 2x para ~0.5x
  o tamanho do modelo. Requer dependência externa.
- **Activation offloading**: move ativações para RAM durante forward, traz de
  volta no backward. Mais lento, mas libera VRAM.
- **KV-cache**: não reduz VRAM de treino, mas acelera geração. Requer
  refatorar o forward para aceitar cache.
