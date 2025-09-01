import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# [checklist.chk, consolidated.00.pth, params.json, pytorch_model.bin, tokenizer.model]
llama3_2_1b_instruct = "../meta-llama/llama-3.2-1b-instruct-files/"
model_state_dict = torch.load(llama3_2_1b_instruct + "consolidated.00.pth", map_location="cpu")

# params
with open(llama3_2_1b_instruct + "params.json", "r") as f: config = json.load(f)
print(json.dumps(config, indent=2))

print("type:", type(model_state_dict))
print("len:", len(model_state_dict))
for k, v in model_state_dict.items(): print(k, v.shape)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kv_dim, i):
        super().__init__()

        self.num_heads = num_heads
        self.num_heads_kv = num_heads / 3
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = model_state_dict[f"layers.{i}.attention.wq.weight"]
        self.k_proj = model_state_dict[f"layers.{i}.attention.wk.weight"]#.T
        self.v_proj = model_state_dict[f"layers.{i}.attention.wv.weight"]#.T
        self.o_proj = model_state_dict[f"layers.{i}.attention.wo.weight"]

    def forward(self, x):
        None

class FF(nn.Module):
    def __init__(self, embed_dim, ff_dim, i):
        super().__init__()
        self.gate_proj = model_state_dict[f"layers.{i}.feed_forward.w1.weight"]#.T
        self.up_proj = model_state_dict[f"layers.{i}.feed_forward.w2.weight"]#.T
        self.down_proj = model_state_dict[f"layers.{i}.feed_forward.w3.weight"]#.T

    def forward(self, x):
        gate_x = F.gelu(x @ self.gate_proj)
        up_x = x @ self.up_proj
        x = gate_x * up_x
        x = x @ self.down_proj
        return x

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads, kv_dim, i):
        super().__init__()

        self.self_attn = SelfAttention(embed_dim, num_heads, kv_dim, i)
        self.input_layernorm = model_state_dict[f"layers.{i}.attention_norm.weight"]
        self.post_attention_layernorm = model_state_dict[f"layers.{i}.ffn_norm_weight.weight"]
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
    def __init__(self):
        super().__init__()

    def forward(self, x):
        None

vocab_size = config["vocab_size"]
embed_dim = config["dim"]
num_layers = config["n_layers"]
ff_dim = embed_dim * config["ffn_dim_multiplier"]
num_heads = config["n_heads"]
kv_dim = 512 # idk yet

