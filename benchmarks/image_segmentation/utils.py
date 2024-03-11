import torch
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    def __init__(self, channels_in=1, channels_out=3, shape=(128, 128, 128),
                 device="cpu", layout="NCDHW", scalar=False):
        shape = tuple(shape)
        x_shape = (channels_in,) + shape if layout == "NCDHW" else shape + (channels_in,)
        self.x = torch.rand((32, *x_shape), dtype=torch.float32, device=device, requires_grad=False)
        if scalar:
            self.y = torch.randint(low=0, high=channels_out - 1, size=(32, *shape), dtype=torch.int32,
                                   device=device, requires_grad=False)
            self.y = torch.unsqueeze(self.y, dim=1 if layout == "NCDHW" else -1)
        else:
            y_shape = (channels_out,) + shape if layout == "NCDHW" else shape + (channels_out,)
            self.y = torch.rand((32, *y_shape), dtype=torch.float32, device=device, requires_grad=False)

    def __len__(self):
        return 64

    def __getitem__(self, idx):
        return self.x[idx % 32], self.y[idx % 32]


