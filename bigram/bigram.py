import torch 

with open("input.txt", 'r') as f:
    text = f.read()


chars = sorted(list(set(text)))

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
index = 0.9*(len(data))
train = data[:n]
test  = data[n:]

