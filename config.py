import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds_path = {"train_ds" : "DataSet/StyleData"}
batch_size = 16
img_h = 64
img_w = 192
min_w = 6
min_h = 2
emb_size = 512
rnn_layers = 2
img_channel = 1
learning_rate = 2e-4
betas = (0.5, 0.999)
epochs = 1000