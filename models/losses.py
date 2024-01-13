import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

import monai


if __name__ == "__main__":
    
    img1 = Variable(torch.rand(1, 1, 256, 256, 256))
    img2 = Variable(torch.rand(1, 1, 256, 256, 256))

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()


    x = torch.ones([1,1,10,10,10])/2
    y = torch.ones([1,1,10,10,10])/2
    print(1-monai.losses.ssim_loss.SSIMLoss(spatial_dims=3)(img1,img2))