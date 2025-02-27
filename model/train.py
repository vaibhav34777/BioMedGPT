
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import tiktoken
import time
import math
import inspect
import torch.utils
import torch._dynamo
torch._dynamo.config.suppress_errors = True

dropout=0.6

@dataclass
class GPTConfig:
    n_embd : int = 768
    n_layer : int =12
    block_size : int = 1024
    vocab_size : int =50304  # converting to a nice number
    n_head : int = 12

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        assert config.n_embd%config.n_head==0
        super().__init__()
        self.c_attn=nn.Linear(config.n_embd,3*config.n_embd)
        #self.register_buffer('tril',torch.tril(torch.ones(config.block_size,config.block_size)).unsqueeze(0).unsqueeze(0))
        # for regularising and making the n_head a batch dimension
        self.n_head=config.n_head
        self.n_embd=config.n_embd
        self.c_proj=nn.Linear(config.n_embd,config.n_embd)
        self.c_proj.NANO_GPT=1
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        B,T,C=x.shape
        qkv=self.c_attn(x)
        q,k,v=qkv.split(self.n_embd,dim=2)
        q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)  # (B,N_HEAD,T,C)
        k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)  # (B,N_HEAD,T,C)
        v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)  # (B,N_HEAD,T,C)
        # wei=q @ k.transpose(2,3)   # (B,N_HEAD,T.T)
        # wei=wei.masked_fill(self.tril[:,:,:T,:T]==0,float('-inf'))
        # wei=F.softmax(wei,dim=-1)
        # y=wei @ v  # (B,N_HEAD,T,C)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True) # flash attention
        y=y.transpose(1,2).contiguous().view(B,T,C)  # (B,T,C)
        y=self.c_proj(y)
        y=self.dropout(y)
        return y

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu=nn.GELU(approximate='tanh')
        self.c_proj=nn.Linear(4*config.n_embd,config.n_embd)
        self.c_proj.NANO_GPT=1
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        x=self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd)
        self.attn=CausalSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.transformer= nn.ModuleDict(dict(
            wte =nn.Embedding(config.vocab_size,config.n_embd),
            wpe =nn.Embedding(config.block_size,config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)
        self.lm_head.weight=self.transformer.wte.weight          # weight sharing
        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            std=0.02
            if hasattr(module,'NANO_GPT'):
                std*=(2*self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self,idx,targets=None):
        B,T=idx.shape
        assert T<=self.config.block_size
        tok_emb=self.transformer.wte(idx)
        pos=torch.arange(0,T,dtype=torch.long,device=idx.device)
        pos_emb=self.transformer.wpe(pos)
        x=tok_emb + pos_emb
        # forward pass
        for block in self.transformer.h:
            x=block(x)
        x=self.transformer.ln_f(x)
        logits=self.lm_head(x)
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,self.config.vocab_size),targets.view(-1))
        return logits,loss
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model



    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


class Dataloader:
    def __init__(self,B,T,split):
        self.B=B
        self.T=T
        with open('pubmedqa.txt','r') as f:
            data=f.read()
        n1=int(0.98*len(data))
        n2=int(0.99*len(data))
        if split=='train':
          text=data[:n1]
        elif split=='validation':
          text=data[n1:n2]
        enc=tiktoken.get_encoding("gpt2")
        tokens=enc.encode(text,allowed_special={"<|endoftext|>"})
        self.tokens=torch.tensor(tokens)
        print(f'no. of tokens {len(self.tokens)}')
        print(f'1 epoch {len(self.tokens)//(B*T)}')
        self.current_postion=0

    def next_batch(self):
        buf=self.tokens[self.current_postion:self.current_postion+self.B*self.T+1]
        x=buf[:-1].view(self.B,self.T)
        y=buf[1:].view(self.B,self.T)
        self.current_postion+=self.B*self.T
        if self.current_postion+self.B*self.T+1<len(self.tokens):
            self.current_postion=0
        return x,y

device='cpu'
if torch.cuda.is_available():
    device='cuda'
torch.manual_seed(1445)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1447)
torch.set_float32_matmul_precision('high')
scaler = torch.cuda.amp.GradScaler()
# MODEL INTIALIZATION
model=GPT.from_pretrained('gpt2')
model.to(device)
model=torch.compile(model)
total_batch_size=65536
B=16
T=128
assert total_batch_size%(B*T)==0
grad_accum_steps=total_batch_size//(B*T)
print(f'grad_accum_steps {grad_accum_steps}')
max_lr=1e-5
min_lr=0.01*max_lr
max_steps=274
warm_up=10
def get_lr(it):
    # warm up stage
    if it<warm_up:
        return max_lr*(it+1)/warm_up
    # ending of decay
    if it>max_steps:
        return min_lr
    # cosine decay after warmup
    decay_ratio=(it-warm_up)/(max_steps-warm_up)
    assert 0<=decay_ratio<=1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0 using cosine decay
    return min_lr + coeff * (max_lr - min_lr)


train_loader=Dataloader(B,T,'train')
val_loader=Dataloader(B,T,'validation')
#optimizer=torch.optim.AdamW(model.parameters(),lr=max_lr,betas=(0.9,0.95),eps=1e-8)  # like gpt3
optimizer=model.configure_optimizers(0.3,max_lr,device)  # using a weight decay of 0.3 acting as regulariser
val_step=20
lossi,val_lossi=[],[]

# TRAINING LOOP
for step in range(max_steps*6):
    loss_accum=0
    t0=time.time()
    optimizer.zero_grad()
    for micro_step in range(grad_accum_steps):
        x,y=train_loader.next_batch()
        x,y=x.to(device),y.to(device)
        with torch.cuda.amp.autocast():
          logits,loss=model(x,y)
          loss=loss/grad_accum_steps
        loss_accum+=loss.detach()
        scaler.scale(loss).backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0) # clipping the gradients to achieve a unit norm
    lr=get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr']=lr
    optimizer.step()
    t1=time.time()
    lossi.append(loss_accum)
    tokens_per_sec=(train_loader.B*train_loader.T*grad_accum_steps)/(t1-t0)
    dt=(t1-t0)*1000
    if step%val_step==0:
      with torch.no_grad():
        x_val,y_val=val_loader.next_batch()
        x_val,y_val=x_val.to(device),y_val.to(device)
        val_logits,val_loss=model(x_val,y_val)
        val_lossi.append(val_loss.item())
        print(f'validation loss at {step} : {val_loss.item():.4f}')
    if step%5==0:
      print(f'step: {step} | loss: {loss_accum:.4f} | norm: {norm} | time: {dt:.4f} ms | tokens_per_sec={tokens_per_sec:.2f}')
