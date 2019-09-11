import torch
from torch import nn
import torch.nn.functional as F
import cv2
import numpy as np

def generate_homo_grid(homo, size):
    #assert type(size) == torch.Size
    N, C, H, W = size

    base_grid = homo.new(N, H, W, 3)
    linear_points = torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1])
    base_grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(base_grid[:, :, :, 0])
    linear_points = torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])
    base_grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(base_grid[:, :, :, 1])
    base_grid[:, :, :, 2] = 1
    grid = torch.bmm(base_grid.view(N, H * W, 3), homo.transpose(1, 2))
    grid = grid.view(N, H, W, 3)
    grid[:, :, :, 0] = grid[:, :, :, 0] / grid[:, :, :, 2]
    grid[:, :, :, 1] = grid[:, :, :, 1] / grid[:, :, :, 2]
    grid = grid[:, :, :, :2].float()
    return grid


def param2theta(keys, w, h):
    #x1, y1, x2, y2, x3, y3, x4, y4 = keys
    src_pts = np.float32([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    dst_pts = np.float32([[-1 + 2 * _x / (w - 1), -1 + 2 * _y / (h - 1)] for (_x, _y) in keys])

    homo, status = cv2.findHomography(src_pts.astype(np.float32), dst_pts.astype(np.float32))
    return homo

class HomoTransfromation(nn.Module):
    def __init__(self, height=32, width=128):
        super(HomoTransfromation, self).__init__()
        self.height = height
        self.width = width
        #self.homo = homo

    def forward(self, input, keys):

        N, C, H, W = input.shape
        size = [N, C, self.height, self.width]
        homos = []
        for i in range(N):
            key = keys[i]
            homo = param2theta(key, W, H)
            homos.append(homo)

        _homo = torch.tensor(homos)
        homo_grid = generate_homo_grid(_homo, size)
        out = F.grid_sample(input, homo_grid)
        return out

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp[0].numpy().transpose((1, 2, 0))
    cv2.imwrite('demo.jpg', inp)
    return

if __name__=='__main__':
    img = cv2.imread('demo/045-114_61-254&414_509&588-495&589_267&486_263&427_490&529-0_0_29_24_4_29_27-123-52.jpg')
    img = img[np.newaxis, :, :, :]
    img = img.transpose([0, 3, 1, 2])
    input = torch.from_numpy(np.array(img, dtype=np.float32))

    keys = np.array([[[263, 427], [490, 529], [495, 589], [267, 486]]], dtype=np.float32)
    print(keys.shape)

    perpective = HomoTransfromation(height=32, width=128)
    out = perpective(input, keys)

    print(out)
    convert_image_np(out)


