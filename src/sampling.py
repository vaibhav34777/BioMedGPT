import tiktoken
import torch
import torch.nn.functional as F
from model import GPT

checkpoint = torch.load('checkpoint_step1050', map_location='cpu')
new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
model=GPT(GPTConfig())
model.load_state_dict(new_state_dict, strict=False)

num_sequence=1
max_tokens=200
enc=tiktoken.get_encoding('gpt2')
tokens=enc.encode("Question: Is myocardial infarct-sparing effect of adenosine A2A receptor activation due to its action on CD4+ T lymphocytes?")
tokens=torch.tensor(tokens,dtype=torch.long)
tokens=tokens.unsqueeze(0).repeat(num_sequence,1)
# predicting the next token
while tokens.shape[1]<max_tokens:
    with torch.no_grad():
        logits,loss=model(tokens)
        logits=logits[:,-1,:]
        probs=F.softmax(logits,dim=-1) 
        topk_probs,topk_indices=torch.topk(probs,50,dim=-1)  # storing top 50 probs and their indexes for the batch
        idx=torch.multinomial(topk_probs,num_samples=1)        # will return a index b/w 0 to 50
        xcol=torch.gather(topk_indices,-1,idx)  # gathering the original indexes using predicted idx and return a tensor of next indexes
        if xcol[-1]==50256:
            break
        tokens=torch.cat((tokens,xcol),dim=1)
# decoding
for i in range(num_sequence):
    x=tokens[i,:max_tokens].tolist()
    decoded=enc.decode(x)
    decode=decoded.split('?')
    print(decode[0])
    print(decode[1])
