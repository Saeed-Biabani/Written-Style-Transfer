import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def display(inp):
    com, tg, tgl = inp

    plt.title(f"target label:{tgl}")
    plt.axis("off")
    
    plt.subplot(1, 2, 1)
    plt.imshow(com, cmap = "gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(tg, cmap = "gray")
    plt.axis("off")
    plt.savefig("fig.png")

def SaveGrid(imgs, fname):
    for i, img in enumerate(imgs):
        img = img.cpu().detach().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img + 1.0) * 127.5
        img = img.astype(np.uint8)
        plt.subplot(4, 4, i+1)
        plt.imshow(img, cmap = "gray")
        plt.axis("off")

    path_ = os.path.join("log", "plots", fname)
    
    plt.savefig(path_)
    plt.close()

def diffmapsave(generated, target ,fname):
    for i in range(16):
      diff_map = torch.sum(torch.abs(generated - target)[i], 0)
      diff_map -= diff_map.min()
      diff_map /= diff_map.max()
      diff_map *= 255
      diff_map = diff_map.detach().cpu().numpy().astype('uint8')
      plt.subplot(4, 4, i+1)
      plt.imshow(diff_map, cmap = "turbo")
      plt.axis("off")
    path_ = os.path.join("log", "plots", fname)  
    plt.savefig(path_)
    plt.close()

def printParams(model, text):
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        params_num.append(np.prod(p.size()))
    print(text.format(sum(params_num)))