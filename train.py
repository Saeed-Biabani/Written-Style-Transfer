from src.utils.DataProvider import DataGenerator
from src.utils.misc import SaveGrid, diffmapsave
from src.utils.DataProvider import DataGenerator
from src.nn.Discriminator import Discriminator
from src.utils.transforms import GanTransforms
from torch.utils.data import DataLoader
from src.utils.misc import printParams
from src.utils.Losses import *
from src.nn.Unet import CUnet
import config as cfg
import torchvision
import torch
import tqdm

def printConfigVars(module, fname):
    pa = [item for item in dir(module) if not item.startswith("__")]
    for item in pa:
        value = eval(f'{fname}.{item}')
        if str(type(value)) not in ("<class 'module'>", "<class 'function'>"):
            print(f"{fname}.{item} : {eval(f'{fname}.{item}')}")

device = cfg.device
printConfigVars(cfg, 'cfg')

ds = DataGenerator(
    root = cfg.ds_path["train_ds"],
    transforms = GanTransforms((cfg.img_h, cfg.img_w))
); ldr = DataLoader(ds, cfg.batch_size, True)

gen = CUnet(cfg).to(device)
printParams(gen, "Mode Parameters : {:,}")
gen_opt = torch.optim.Adam(gen.parameters(), lr = cfg.learning_rate)

dis = Discriminator(1).to(device)
dis_opt = torch.optim.Adam(dis.parameters(), lr = cfg.learning_rate)

intensity_loss = IntensityLoss(True)
gradient_loss = GradientLoss(1, True)
adversarial_loss = GeneratorAdversarialLoss(True)
discriminator_loss = DiscriminatorAdversarialLoss(True)

for epoch in range(cfg.epochs):
    loop = tqdm.tqdm(ldr, colour = "green")
    for batch_indx, (img, tg) in enumerate(loop):
        img = img.to(device)
        tg = tg.to(device)
        
        gen_img = gen(img)
        
        dis_opt.zero_grad()
        d_fake = dis(gen_img)
        d_real = dis(img)
        
        int_loss = intensity_loss(gen_img, tg)
        grad_loss = gradient_loss(gen_img, tg)
        adv_loss = adversarial_loss(d_fake)
        
        g_loss = int_loss + grad_loss + 0.05 * adv_loss
        
        d_loss = discriminator_loss(d_real, d_fake)
        
        dis_opt.zero_grad()
        d_loss.backward(retain_graph=True)
        gen_opt.zero_grad()
        g_loss.backward()
        dis_opt.step()
        gen_opt.step()
        
        loop.set_postfix({
            "epoch" : epoch+1,
            "d_loss" : d_loss.item(),
            "g_loss" : g_loss.item()
        })
        
        if batch_indx % 500 == 0:
            with torch.no_grad():
                fake = gen(img)
                
                fake_grid = torchvision.utils.make_grid(
                    [fake[:16]], normalize = True
                )
                SaveGrid(fake_grid, f"images/plot_fake_{epoch+1}_{batch_indx+1}.png")
                diffmapsave(fake, tg, f"hitmap/plot_fake_{epoch+1}_{batch_indx+1}_hmap.png")
    torch.save(gen.state_dict(), "textStyle.pth")