import streamlit as st
import torch
import requests
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import tiktoken
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# ---------------- Model Definition ---------------- #

@dataclass
class GPTConfig:
    n_embd: int = 1024
    n_layer: int = 24
    block_size: int = 1024
    vocab_size: int = 50257  
    n_head: int = 16

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        assert config.n_embd % config.n_head == 0
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANO_GPT = 1

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANO_GPT = 1
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # weight sharing
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANO_GPT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size
        tok_emb = self.transformer.wte(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        return logits, loss

# ---------------- Load Model from Hugging Face ---------------- #
def download_model():
    model_path = "model/model.pth"
    if not os.path.exists(model_path):
        os.makedirs("model", exist_ok=True)
        url = "https://huggingface.co/imvaibhavrana/bio-med-gpt/resolve/main/checkpoint_step1050"
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)
    return model_path
model_path = download_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(GPTConfig()).to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# ---------------- Inference Function ---------------- #

def generate_text(prompt, max_tokens=128, topk=50):
    # Prepend "Question:" and append "?" automatically.
    if not prompt.lower().startswith("question:"):
        prompt = "Question: " + prompt
    if not prompt.strip().endswith("?"):
        prompt = prompt.strip() + "?"
    
    # Use tiktoken GPT-2 encoding to tokenize input
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    while tokens.shape[1] < max_tokens:
        with torch.no_grad():
            logits, _ = model(tokens)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, topk, dim=-1)
            next_token_index = torch.multinomial(topk_probs, num_samples=1)
            next_token = torch.gather(topk_indices, -1, next_token_index)
            if next_token.item() == 50256:
                break
            tokens = torch.cat((tokens, next_token), dim=1)
    
    generated_text = enc.decode(tokens[0].tolist())
    # Extract and return only the part after "Answer:"
    if "Answer:" in generated_text:
        return generated_text.split("Answer:")[-1].strip()
    else:
        return generated_text.strip()

# ---------------- Streamlit App Layout ---------------- #

st.set_page_config(page_title="BioMedGPT", page_icon="🤖", layout="wide")
st.title("BioMedGPT – Biomedical Q&A Model")

st.markdown("""
Welcome to the BioMedGPT interactive demo.  
Enter your biomedical question below.  
*Your input will automatically be prefixed with "Question:" and suffixed with "?"*.  
The generated response will display only the answer section (text after "Answer:").
""")

user_input = st.text_input("Enter your biomedical question:")

if st.button("Generate Answer") and user_input:
    with st.spinner("Generating answer..."):
        answer = generate_text(user_input, max_tokens=128)
    st.markdown("### Generated Answer:")
    st.write(answer)
