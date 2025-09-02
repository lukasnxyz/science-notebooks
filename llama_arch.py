import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from transformers import AutoTokenizer

# [checklist.chk, consolidated.00.pth, params.json, pytorch_model.bin, tokenizer.model]
llama3_2_1b_instruct = "../meta-llama/llama-3.2-1b-instruct-files/"
model_state_dict = torch.load(llama3_2_1b_instruct + "consolidated.00.pth", map_location="cpu")

# params
with open(llama3_2_1b_instruct + "params.json", "r") as f: config = json.load(f)
print(json.dumps(config, indent=2))

#print("type:", type(model_state_dict))
#print("len:", len(model_state_dict))
#for k, v in model_state_dict.items(): print(k, v.shape)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, i):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = model_state_dict[f"layers.{i}.attention.wq.weight"]
        self.k_proj = model_state_dict[f"layers.{i}.attention.wk.weight"].T
        self.v_proj = model_state_dict[f"layers.{i}.attention.wv.weight"].T
        self.o_proj = model_state_dict[f"layers.{i}.attention.wo.weight"]

    def forward(self, x):
        bsz, seq_len, embed_dim = x.size()

        # Project Q, K, and V
        q = torch.matmul(x, self.q_proj)  # (bsz, seq_len, num_heads * head_dim)
        k = torch.matmul(x, self.k_proj)  # (bsz, seq_len, num_heads_kv * head_dim)
        v = torch.matmul(x, self.v_proj)  # (bsz, seq_len, num_heads_kv * head_dim)

        # Transpose to get (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ v

        # Combine heads and pass through output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, embed_dim)
        return torch.matmul(attn_output, self.o_proj)


class FF(nn.Module):
    def __init__(self, embed_dim, ff_dim, i):
        super().__init__()
        self.gate_proj = model_state_dict[f"layers.{i}.feed_forward.w1.weight"].T
        self.up_proj = model_state_dict[f"layers.{i}.feed_forward.w2.weight"]#.T
        self.down_proj = model_state_dict[f"layers.{i}.feed_forward.w3.weight"]#.T

    def forward(self, x):
        gate_x = F.gelu(x @ self.gate_proj)
        up_x = x @ self.up_proj
        x = gate_x * up_x
        x = x @ self.down_proj
        return x

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads, i):
        super().__init__()

        self.self_attn = SelfAttention(embed_dim, num_heads, i)
        self.input_layernorm = model_state_dict[f"layers.{i}.attention_norm.weight"]
        self.post_attention_layernorm = model_state_dict[f"layers.{i}.ffn_norm.weight"]
        self.mlp = FF(embed_dim, ff_dim, i)
        self.eps = 1e-5

    def forward(self, x):
        residual = x

        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(variance + self.eps)

        x = self.self_attn(x) + residual

        residual = x

        mean = x.mean(dim=-1, keepdim=True)  # last layer normalization
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        x = self.input_layernorm * x

        x = self.mlp(x) + residual
        return x

class Llama3_2(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, ff_dim, num_heads):
        super().__init__()

        self.embed_tokens = model_state_dict["tok_embeddings.weight"]
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, ff_dim, num_heads, i)
            for i in range(num_layers)
        ])
        self.norm = model_state_dict["norm.weight"]
        self.lm_head = model_state_dict["output.weight"].T # final layer
        self.eps = 1e-5

    def forward(self, sentence):
        tokenized_output = tokenizer(sentence, return_tensors="pt")
        input_ids = tokenized_output["input_ids"]

        x = (self.embed_tokens)[input_ids]

        # Pass through each layer
        for layer in self.layers:
            x = layer(x)

        # Normalize output
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(variance + self.eps)  # Layer normalization
        x = x * self.norm

        # Project to vocabulary size
        logits = x @ self.lm_head
        return logits




vocab_size = config["vocab_size"]
embed_dim = config["dim"]
num_layers = config["n_layers"]
ff_dim = embed_dim * config["ffn_dim_multiplier"]
num_heads = config["n_heads"]

model = Llama3_2(vocab_size, embed_dim, num_layers, ff_dim, num_heads)

sentence = "my name is "

logits = model.forward(sentence)

print("logits shape:", logits.shape)

token_ids = torch.argmax(logits, dim=-1)
decoded_tokens = [tokenizer.decode(ids, skip_special_tokens=True) for ids in token_ids]

print(f"token ids: {token_ids}")
print(f"decoded tokens: {decoded_tokens}")

