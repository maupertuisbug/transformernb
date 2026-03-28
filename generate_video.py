import gymnasium as gym
import cv2 
import os 
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
import imageio
from sklearn.feature_extraction import image
import torch
import numpy as np

env = gym.make('Humanoid-v5', render_mode='rgb_array')

imgs = [] 

obs, _ = env.reset()
frame  = env.render()
imgs.append(frame[:256, :256, :])
for _ in range(0, 1000):
    action = env.action_space.sample()
    next_obs, reward, _, _, info = env.step(action)
    frame = env.render()
    imgs.append(frame[:256, :256, :])
env.close()
    
frame = imgs[0]
h, w, c = frame.shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter('test.mp4', fourcc, 20.0, (w, h))

for frame in imgs:
    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

cv2.destroyAllWindows()
video.release()

frames = imageio.mimread("test.mp4")

imageio.mimsave("output.gif", frames, fps=10)
t = 0 
for frame in imgs:
    path = os.path.join('imgs/',f"frame_{t}.png")
    cv2.imwrite(path, frame)
    t+=1


frame = imgs[0]
patch_size = 16
patch_sq = patch_size*patch_size
patches = (256*256)//patch_sq
# for h_ in range(0, h-patch_size+1, patch_size):
#     for w_ in range(0, w-patch_size+1, patch_size):
#         p = frame[h_:h_+patch_size, w_:w_+patch_size]
#         patches.append(p.flatten())


# frame_seq = imgs[:32]
# frame_s = [torch.tensor(np.ascontiguousarray(x), dtype=torch.float).reshape(patches, patch_sq*3) for x in frame_seq]
# frame_stack = torch.stack(frame_s, dim = 0)
# out = frame_stack
# s, l, w = frame_stack.shape
# linear_l = torch.nn.Linear(w, 512)
# pos_embedding = torch.nn.Embedding(s*l, 512)

# out  = out.reshape(s*l, w)
# out  = linear_l(out) # s*l , 512
# ps  = pos_embedding(torch.arange(0, s*l))
# out = out+ps
# print(out.shape)


