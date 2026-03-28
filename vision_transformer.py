import torch 
from tqdm import tqdm 
import gymnasium as gym
import cv2 
import os 
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
import imageio
from sklearn.feature_extraction import image
import torch
import numpy as np
import glob
from decoder_transformer.llm_heads import SingleHead, FeedForward

params = {

    'block_size' : 32,
    'd' : 128, 
    'n_embed' : 384, 
    'n_heads' : 6,
    'patch_size' : 16,
    'batch_size' : 32 
}

class PatchEmbedding(torch.nn.Module):

    def __init__(self, block_size, patches, patch_sq, d):

        super().__init__()

        self.b = block_size 
        self.l = patches 
        self.w = patch_sq
        self.linear_layer = torch.nn.Linear(patch_sq*3, d)
        self.pos_embedding = torch.nn.Embedding(block_size*patches, d)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        b, s, l, w = x.shape
        out = x.reshape(b, s*l, w)
        out = self.linear_layer(out)
        ps  = self.pos_embedding(torch.arange(0, self.b*self.l).to(self.device))

        out = out + ps 
        return out

class VisionTransformer(torch.nn.Module):

    def __init__(self, params):

        super().__init__()
        self.block_size = int(params['block_size'])
        self.d    = int(params['d'])
        self.n_embed = int(params['n_embed'])
        self.batch_size = int(params['batch_size'])
        self.n_heads    = int(params['n_heads'])
        self.patch_size = int(params['patch_size'])
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        self.patch_sq = self.patch_size*self.patch_size
        self.patches = (256*256)//self.patch_sq
        self.data = self.get_data()

        n = int(0.8*len(self.data))
        self.train_data = self.data[:n]
        self.test_data = self.data[n:]


        self.patch_embedding = PatchEmbedding(self.block_size, self.patches, self.patch_sq, self.d).to(self.device)
        self.patch_lm        = torch.nn.Linear(self.d, self.n_embed).to(self.device)
        self.lm_head         = torch.nn.Linear(self.n_embed, self.patch_sq*3).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0004)


    def forward(self, x, targets=None):

        batch, block, seq, c = x.shape # batch, block, seq, c 
        out = self.patch_embedding(x)
        out = self.patch_lm(out)
        out = self.lm_head(out)
        out = out.reshape(batch, block, seq, self.patch_sq*3)

        fun = torch.nn.MSELoss()
        if targets == None:
            loss = None
        else:
            loss = fun(out, targets)

        return out, loss 


    def get_data(self):

        imgs = [] 
        patches = self.patches
        patch_sq = self.patch_sq

        for filepath in glob.glob(os.path.join("imgs", '*.png')):

            brg_img = cv2.imread(filepath)

            rgb_img = cv2.cvtColor(brg_img, cv2.COLOR_BGR2RGB)

            imgs.append(torch.tensor(np.ascontiguousarray(rgb_img/255.0), dtype=torch.float).reshape(patches, patch_sq*3))

        return imgs

    
    def get_train_batch(self):

        data = self.train_data # [(l, w),..]
        idx = torch.randint(len(self.data)-self.block_size, (self.batch_size,))

        data_x = [torch.stack(data[x:x+self.block_size], dim=0) for x in idx]
        data_x = torch.stack(data_x, dim=0).to(self.device)

        data_y = [torch.stack(data[x+1:x+1+self.block_size], dim=0) for x in idx] # (batch_size, block_size, l, w)
        data_y = torch.stack(data_y, dim=0).to(self.device)

        return data_x, data_y


    def learn(self, epochs):

        for iter in tqdm(range(0, epochs)):

            x, y = self.get_train_batch()

            logits, loss = self(x, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Train Loss :", loss)

    
    def generate(self, seq, max_frames, block):

        seq_p = torch.stack(seq[-block:], dim=0).to(self.device)
        seq = torch.stack(seq, dim=0).to(self.device)
        for _ in range(0, max_frames):

            logits, loss = self(seq_p.unsqueeze(0))
            logits = logits[:, -1, :, :]

            seq = torch.cat([seq, logits], dim=0)
            seq_p = seq[-block:,:,:]

        h, w, c = 256, 256, 3
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter('test.mp4', fourcc, 20.0, (w, h))

        for frame in range(0, seq.shape[0]):
            img = (seq[frame]*255.0).to(torch.uint8)
            video.write(cv2.cvtColor(img.reshape(256, 256, 3).detach().cpu().numpy(), cv2.COLOR_RGB2BGR))

        cv2.destroyAllWindows()
        video.release()

        frames = imageio.mimread("test.mp4")
        imageio.mimsave("output.gif", frames, fps=10)




vt = VisionTransformer(params)
vt.learn(30)

env = gym.make('Humanoid-v5', render_mode='rgb_array')

imgs = [] 
patches = (256*256)//256
patch_sq = 256

obs, _ = env.reset()
frame  = env.render()
bs = 32
for i in range(0,bs+5):
    imgs.append(torch.tensor(np.ascontiguousarray(frame[:256, :256, :]), dtype=torch.float, device = vt.device).reshape(patches, patch_sq*3))

vt.generate(imgs, 100, bs)




    

    

