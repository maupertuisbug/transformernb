import torch
from tqdm import tqdm


params = {

    'block_size' : 32, 
    'n_embed' : 16,

}


class Decoder(torch.nn.Module):

    def __init__(self, params):
        super().__init__()
        
        self.text = None 
        self.block_size = params['block_size']
        self.n_embed = int(params['n_embed'])
        self.get_text()
        self.data = self.encode(self.text)
        n = int(0.7 * len(self.data))
        self.train = self.data[:n]
        self.test = self.data[n:]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding = torch.nn.Embedding(self.vocab_size, self.n_embed).to(self.device)
        self.lm_head   = torch.nn.Linear(self.n_embed, self.vocab_size).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.006)

    def get_text(self):

        with open("input.txt", 'r') as f:
            self.text = f.read()

        chars = sorted(list(set(self.text)))
        self.vocab_size = len(chars)

        map_char = {ch:i for i, ch in enumerate(chars)}
        inv_map  = {i:ch for i, ch in enumerate(chars)}

        self.encode = lambda l : [map_char[c] for c in l]
        self.decode = lambda l : ''.join([(inv_map[i]) for i in l])

    def get_batch(self, batch_size):

        idx = torch.randint(len(self.train) - self.block_size, (batch_size,))

        data_x = torch.stack([torch.tensor(self.train[x:x+self.block_size]) for x in idx])
        data_y = torch.stack([torch.tensor(self.train[x+1:x+1+self.block_size]) for x in idx])

        return data_x.to(self.device), data_y.to(self.device)

    def get_val_batch(self, batch_size):

        idx = torch.randint(len(self.test) - self.block_size, (batch_size,))

        data_x = torch.stack([torch.tensor(self.train[x:x+self.block_size]) for x in idx])
        data_y = torch.stack([torch.tensor(self.train[x+1:x+1+self.block_size]) for x in idx])

        return data_x.to(self.device), data_y.to(self.device)

    def loss(self, logits, targets):

        if targets == None:
            l = None 

        else :
            loss = torch.nn.CrossEntropyLoss()
            b, t, c = logits.shape
            logits = logits.view(b*t, c)
            targets = targets.view(b*t)
            l = loss(logits, targets)

        return l

    def forward(self, x, targets = None):

        x = self.embedding(x)
        logits = self.lm_head(x)

        loss = self.loss(logits, targets)

        return logits, loss

    def validate(self):
          
        x, y = self.get_val_batch(batch_size=64)
        logits, l = self(x, y)

        return l

    def learn(self, epochs):

        for iter in tqdm(range(0, epochs)):
            x, y = self.get_batch(batch_size=64)
            logits, l = self(x, y)
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()

        train_loss = l 
        val_loss = self.validate()
        print("Train Loss :", train_loss)
        print("Val Loss :", val_loss)

    def generate(self, vec, max_length):

        for _ in range(0, max_length):

            logits, loss = self(vec)
            logits = logits[:, -1, :]
            logits = torch.nn.functional.softmax(logits, dim=-1)

            vec_next = torch.multinomial(logits, 1)
            vec = torch.cat([vec, vec_next], dim=1)

        return vec



model = Decoder(params)
model.learn(epochs=10000)
input_vec = torch.zeros(size=(1, 1), dtype = torch.long, device = model.device)
output_vec = model.generate(input_vec, 1000)
output = output_vec.squeeze(0).tolist()
print(model.decode(output))

        