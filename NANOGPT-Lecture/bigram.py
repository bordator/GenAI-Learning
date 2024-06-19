import torch
import torch.nn as nn
import torch.nn.functional as F

#globals - hyperparameter
batch_size = 32
block_size = 8
max_iters = 3000
eval_iters = 200
eval_interval = 300
learning_rate = 1e-2
#----------

torch.manual_seed(1337)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


#load the train dataset - tiny shakespeare
#!curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o input.txt

#read the data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#vocab from the text file    
chars = sorted(set(text))
vocab_size = len(chars)

#create a mapping from characters to integers
stoi = { c:i for i, c in enumerate(chars) }
itos = { i:c for i, c in enumerate(chars)}
s : str
encode = lambda s : [ stoi[x] for x in s] #encoder: Take a string and convert it do an integer 
it : list
decode  = lambda it : ''.join([itos[x] for x in it]) #decoder: take a list of integers, returns a string

#split of train and val data
data : torch.tensor = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * (len(data)))
train_data = data[:n]
val_data = data[n:]

def get_batch(split: str) -> tuple[torch.tensor, torch.tensor] : 
    data = train_data if split == 'train' else val_data
    ix : torch.tensor = torch.randint(len(data)-batch_size, (batch_size,))
    x : torch.Tuple[torch.Tensor, ] = torch.stack([data[i:i+block_size] for i in ix ])
    y : torch.Tuple[torch.Tensor, ] =  torch.stack([data[i+1:i+block_size+1] for i in ix ])
    return x, y



class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size : int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #like in makemore each letter and the probality of the next token is stored in the embedding
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size, device=device)
    
    def forward(self, idx, targets=None):
        idx = idx.to(device)
        
        logits = self.token_embedding_table(idx) #B-Batch, T - Time, C - Channel
        #print('logits:', logits)
        #why C - 65 ? This a are the class probalities for a certain character
        #logits has 4B, 8T, 65C <> Targets is 4B, 8T
        
        if(targets is None):
            loss=None
        else:
            targets = targets.to(device)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
    #        print(logits.shape)
    #        print('targets', targets)
            targets = targets.view(B*T)

    #        print(targets.shape)
    #        print(targets)
            loss = F.cross_entropy(logits, targets)
#        print(loss)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #ids is (B,T) array of indicies in the current context
        for i in range(max_new_tokens):
            #predict the next tokens
            logits, loos = self(idx)
            #after prediction select the most proably next token. We use Softmax for prediction
            #focus on the last time step
            #print(f'logits shape: {logits.shape}')
            logits = logits[:,-1,:] #becomes (B,C)
            #print(f'After trans logits shape: {logits.shape}')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            #append sampled index to the running contect
            idx = torch.cat((idx, idx_next), dim=1) #(B, B+1)
        return idx            


model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimzer : torch.optim.AdamW= torch.optim.AdamW(m.parameters(), lr=1e-3)

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    #sample a batch of data
    xb, yb = get_batch('train')
    
    #evaluate the loss
    logits, loss = model(xb, yb)
    optimzer.zero_grad(set_to_none=True)
    loss.backward()
    optimzer.step()
    
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))