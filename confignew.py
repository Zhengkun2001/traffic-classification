import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
epochs = 60
Fold_num = 5

H_indim = 5
P_indim = 128

H_DEPTH = 4
P_DEPTH = 2

OUT_DIM = 128
NUM_HEAD = 8
DROP = 0.2
ATT_MASK = True
H_ATT_MASK = False
P_ATT_MASK = False

