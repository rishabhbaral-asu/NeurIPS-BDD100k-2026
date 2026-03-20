import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    """Modernized Causal Attention with PyTorch 2.0 SDPA."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout_val = dropout

    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate Query, Key, and Value
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)
        
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        # PyTorch 2.0 Flash Attention (Highly optimized, handles causal masking automatically)
        # Note: is_causal=True handles the masking without needing explicit tril matrices
        y = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout_val if self.training else 0.0, 
            is_causal=True
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection with residual dropout
        return self.resid_dropout(self.c_proj(y))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim, bias=False)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim, bias=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MiniLLM(nn.Module):
    """Conference-grade Mini GPT with proper initialization and multimodal support."""
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=4, max_seq_len=128, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)
        
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim, bias=False)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Weight sharing between token embedding and final output projection (standard practice)
        self.token_embedding.weight = self.lm_head.weight

        # Apply custom initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Standard GPT-2 initialization for stable training."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * len(self.blocks)))

    def forward(self, input_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Must provide either input_ids or inputs_embeds")
            inputs_embeds = self.token_embedding(input_ids)
            
        B, T, C = inputs_embeds.size()
        assert T <= self.max_seq_len, f"Cannot forward sequence of length {T}, block size is only {self.max_seq_len}"
        
        # Generate position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=inputs_embeds.device)
        pos_emb = self.position_embedding(pos)
        
        # Combine and apply dropout
        x = self.emb_dropout(inputs_embeds + pos_emb)
        
        # Forward through transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Generate next-token logits
        logits = self.lm_head(x)
        return logits