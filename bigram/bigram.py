import torch 

with open("input.txt", 'r') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

map_char = {ch:i for i,ch in enumerate(chars)}
inv_map  = {i:ch for i,ch in enumerate(chars)}

encode = lambda l : [map_char[c] for c in l]
decode = lambda l : ''.join([(inv_map[i]) for i in l])

encoded = encode("Hello World!")
print(encoded)
decoded = decode(encoded)
print(decoded)

data = encode(text)

# split train and test 
n = int(0.9*(len(data)))
train = data[:n]
test  = data[n:]

block_size = 8

def get_batch(batch_size):
    idx = torch.randint(len(train)-block_size, (batch_size,))

    val_x = torch.stack([torch.tensor(train[x:x+block_size]) for x in idx])
    val_y = torch.stack([torch.tensor(train[x+1:x+block_size+1]) for x in idx])

    return val_x, val_y


class BigramLM(torch.nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, vocab_size)
        self.optimizer = torch.optim.Adam(self.embedding.parameters(), lr = 0.003)

    def forward(self, x, targets = None):

        logits = self.embedding(x)

        loss =  torch.nn.CrossEntropyLoss()
        b, t, c = logits.shape 
        logits = logits.view(b*t, c)
        targets = targets.view(b*t)

        l = loss(logits, targets)
        return logits, l

    def train(self):

        for iter in range(0, 10000):
            x, y = get_batch(batch_size=32)
            logits, l = self(x, y)
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
        
        print(l)

x, y = get_batch(4)
mm = BigramLM(vocab_size)
mm.train()