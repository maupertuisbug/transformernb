import torch 
from encoder_transformer.llm_heads import EncoderBlock


class SingleHead(torch.nn.Module):

    def __init__(self, n_embed, head_size, t):
        super().__init__()
        
        self.key = torch.nn.Linear(n_embed, head_size, bias = False)
        self.query = torch.nn.Linear(n_embed, head_size, bias = False)
        self.value = torch.nn.Linear(n_embed, head_size, bias= False)
        self.layer = torch.nn.LayerNorm(head_size)
        self.t = t

        self.register_buffer('tril', torch.tril(torch.ones(t, t)))

    
    def forward(self, x):

        k = self.key(x)
        q = self.query(x)

        aff = q @ k.transpose(-2, -1)

        aff = aff.masked_fill((self.tril[:self.t, :self.t] == 0), value = float('-inf'))
        aff = torch.nn.functional.softmax(aff, dim = -1)

        out = aff @ self.value(x)

        out = self.layer(out)

        return out

class MultiHead(torch.nn.Module):

    def __init__(self, n_heads, n_embed, head_size, t):

        super().__init__()

        self.net = torch.nn.ModuleList([SingleHead(n_embed, head_size, t) for h in range(0, n_heads)])


    def forward(self, x):

        out = [h(x) for h in self.net]
        out = torch.cat(out, dim=-1)
        out = out + x
        return out

class FeedForward(torch.nn.Module):

    def __init__(self, head_size, n_embed):

        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(head_size, 4*n_embed), 
            torch.nn.ReLU(), 
            torch.nn.Linear(4*n_embed, n_embed),
            torch.nn.LayerNorm(n_embed)
        )

    def forward(self, x):

        out = x  + self.net(x)
        return out

class BlockSH(torch.nn.Module):

    def __init__(self, n_embed, head_size, n_heads, t):

        super().__init__()

        self.net = torch.nn.Sequential(
            SingleHead(n_embed, head_size, t),
            SingleHead(head_size, head_size, t),
            FeedForward(head_size, n_embed)
        )

    def forward(self, x):
        out = self.net(x)
        return out

class BlockMH(torch.nn.Module):

    def __init__(self, n_embed, head_size, n_heads, t):

        super().__init__()

        self.net = torch.nn.Sequential(
            MultiHead(n_heads, n_embed, head_size//n_heads, t),
            MultiHead(n_heads, head_size, head_size//n_heads, t),
            FeedForward(head_size, n_embed)
        )

    def forward(self, x):
        out = self.net(x)
        return out

class Block(torch.nn.Module):

    def __init__(self, n_embed, head_size, n_heads, t):
        super().__init__()

        self.encoder = torch.nn.Sequential(
                EncoderBlock(n_embed, head_size, n_heads, t),
                EncoderBlock(n_embed, head_size, n_heads, t)
        )

        self.head_one = MultiHead(n_heads, n_embed, head_size//n_heads, t)
        self.decoder = torch.nn.Sequential(
                MultiHead(n_heads, 2*head_size, 2*head_size//n_heads, t),
                FeedForward(2*head_size, n_embed)
            )
    
    def forward(self, x):

        out_one = self.head_one(x)
        out_two = self.encoder(x)
        out = torch.cat([out_one, out_two], dim=-1)

        out = self.decoder(out)
        out_one = self.head_one(out)
        out_two = self.encoder(out)
        out = torch.cat([out_one, out_two], dim=-1)

        out = self.decoder(out)

        return out
