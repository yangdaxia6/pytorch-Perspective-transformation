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
    grid = grid.view(N, H, W, 2)
    return grid

def rotate2theta(N, roi, width, height):

    theta = torch.zeros(size=(N, 2, 3))
    theta = torch.autograd.Variable(theta)
    x1, y1, x2, y2 = roi
    theta[:, 0, 0] = (x2 - x1) / (width - 1)
    theta[:, 0 ,2] = (x1 + x2 - width + 1) / (width - 1)
    theta[:, 1, 1] = (y2 - y1) / (height - 1)
    theta[:, 1, 2] = (y1 + y2 - height + 1) / (height - 1)
    return theta

class CropTransfromation(nn.Module):
    def __init__(self, height=32, width=96):
        super(CropTransfromation, self).__init__()
        self.height = height
        self.width = width
        #self.homo = homo

    def forward(self, input, rois):

        N, C, H, W = input.shape
        size = [N, C, self.height, self.width]
        out = torch.rand([N, C, self.height, self.width])
        theta = rotate2theta(N, rois, W, H)
        #homo_grid = F.affine_grid(theta, out.shape)
        homo_grid = generate_homo_grid(theta, out.shape)
        print(homo_grid)
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
    roi = np.array([263, 427, 495, 589], dtype=np.float32)

    rotate = CropTransfromation(height=162, width=232)
    out = rotate(input, roi)
    print(out)
    convert_image_np(out)


