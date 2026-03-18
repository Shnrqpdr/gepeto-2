import torch
import torch.nn as nn
from .layers import TransformerBlock


class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_len, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.context_len = context_len
        self.num_layers = num_layers

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(context_len, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: compartilha pesos entre token embedding e output projection
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        # Scaled residual init (GPT-2): projecoes residuais escaladas por 1/sqrt(2*N)
        residual_std = 0.02 / (2 * num_layers) ** 0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp[2].weight, mean=0.0, std=residual_std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]

            # Repetition penalty: penaliza tokens que ja apareceram
            if repetition_penalty != 1.0:
                for i in range(idx.size(0)):
                    seen = idx[i].unique()
                    penalty_logits = logits[i, seen]
                    logits[i, seen] = torch.where(
                        penalty_logits > 0,
                        penalty_logits / repetition_penalty,
                        penalty_logits * repetition_penalty,
                    )

            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                # Zera tudo apos ultrapassar top_p
                mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[mask] = -float('inf')
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
