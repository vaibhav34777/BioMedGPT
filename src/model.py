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

