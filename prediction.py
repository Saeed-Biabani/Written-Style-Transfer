from src.nn.Unet import CUnet
import config as cfg
import numpy as np
import pathlib
import torch
import cv2

def PrepareCustomImage(fname):
    image = torch.tensor(cv2.resize(cv2.imread(fname, 0), (192, 64)))[None, ...].float()
    return (image - image.mean()) / image.std()

def printConfigVars(module, fname):
    pa = [item for item in dir(module) if not item.startswith("__")]
    for item in pa:
        value = eval(f'{fname}.{item}')
        if str(type(value)) not in ("<class 'module'>", "<class 'function'>"):
            print(f"{fname}.{item} : {eval(f'{fname}.{item}')}")

device = cfg.device

gen = CUnet(cfg).to(device)
gen.load_state_dict(torch.load("path/to/model/weights.pth"))


fname = pathlib.Path("path/to/input/image.png")
img = PrepareCustomImage(str(fname))

pred = gen(img[None, ...].to(device)).detach().cpu().numpy()

generated_image = (pred[0][0] + 1.) * 127.5
input_image = (img[0].cpu().numpy() + 1.) * 127.5
    
eps = np.zeros((64, 2))
res = np.concatenate((input_image, eps, generated_image), axis = 1)

cv2.imwrite(f"generated_{fname.stem}.png", res)