
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import tiktoken
import time
import math
import inspect
import torch.utils
from model import GPT   # importing GPT class from model.py
import torch._dynamo
torch._dynamo.config.suppress_errors = True

dropout=0.3

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
model=GPT.from_pretrained('gpt2-medium')
model.to(device)
model=torch.compile(model)
# Freezibg first 12 transformer blocks out of 24
for block in model.transformer.h[:12]:
    for param in block.parameters():
        param.requires_grad = False
total_batch_size=65536
B=16
T=128
assert total_batch_size%(B*T)==0
grad_accum_steps=total_batch_size//(B*T)
print(f'grad_accum_steps {grad_accum_steps}')
max_lr=1e-6
min_lr=0.01*max_lr
max_steps=548
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
for step in range(max_steps*2):
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
      print(f'step: {step} | loss: {loss_accum:.4f} | time: {dt:.4f} ms | tokens_per_sec={tokens_per_sec:.2f}')
