import torch
import torch.nn.functional as F
import torch._C._fft as fft
import math
import cv2
from typing import List

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def getGaussianHpf2d(shape,sigma=1):
    u = torch.from_numpy(cv2.getGaussianKernel(shape[0], sigma))
    v = torch.from_numpy(cv2.getGaussianKernel(shape[1], sigma))
    # filter = torch.outer(u, v)
    filter = u*v.t()
    filter /= filter.max() 
    return 1-filter

def apofield_mask(shape):
    aporad = int(min(shape)*0.12)
    apos = torch.hann_window(aporad*2, False)
    row = torch.ones(shape[0])
    cols = torch.ones(shape[1])
    row[:aporad] = apos[:aporad]
    row[-aporad:] = apos[-aporad:]
    cols[:aporad] = apos[:aporad]
    cols[-aporad:] = apos[-aporad:]

    # apofield = torch.outer(row, cols)
    apomask = torch.ones(shape, dtype=torch.bool)
    radius = aporad//2
    apomask[radius:-radius, radius:-radius] = 0

    # Returns apofield, apomask
    return torch.outer(row, cols), apomask

def getPolarMap(shape):
    """
    Write log base function directly to map.
    function returns map for log-polar transform given a shape
    Arguments:
        shape: A tuple of int
    Returns:
        A torch Tensor of polarmap normalized to [-1,1]
    """

    logBase = (0.55*shape[0])**(1/shape[1])
    log = torch.pow(logBase, torch.arange(shape[1]))
    ang = -torch.linspace(0, math.pi, shape[0])

    sampleY = torch.outer(ang.sin(), log)+shape[0]/2
    sampleX = torch.outer(ang.cos(), log*shape[1]/shape[0])+shape[1]/2

    return torch.stack([sampleX/(shape[1]-1), sampleY/(shape[0]-1)], 2).to(torch.float).unsqueeze(0)*2-1

def index_generator(shape):
    offset = torch.arange(5).unsqueeze(1)-2
    index = (offset*(shape[1])+offset.T).view([1,-1])
    return index

class MyModule(torch.nn.Module):

    def __init__(self, filterG, img_shape=(180, 240)) -> None:
        super().__init__()
        apofield, apomask = apofield_mask(img_shape)
        self.register_buffer('apofield',apofield)
        self.register_buffer('apomask',apomask)
        self.register_buffer('index',index_generator(img_shape))
        self.register_buffer('polarmap', getPolarMap(img_shape))
        self.register_buffer('filterG', filterG)
        self.register_buffer('col',torch.arange(0, 5, 1).repeat(5))

    def logPolarNBatch(self, imgs: List[torch.Tensor]):
        img = torch.stack(imgs, 0)
        l_img = len(imgs)
        imgApo = img*self.apofield + img.masked_select(self.apomask).view(
            [l_img, -1]).mean(1).view([l_img, 1, 1])*(1-self.apofield)
        imgDftHpf = (fft.fft_fftshift(
            fft.fft_fft2(imgApo), [1, 2])*self.filterG).abs()

        return F.grid_sample(imgDftHpf.unsqueeze(1),
                                self.polarmap.expand([l_img, -1, -1, -1]),
                                mode='bicubic',
                                align_corners=True).squeeze(1)

    def pcorr_rot_batch_1d(self, lp: torch.Tensor):
        f = torch.fft.rfft(lp,dim=1)
        a = torch.nn.functional.normalize(f[0]*f[1].conj())
        b = torch.fft.irfft(a,dim=0).sum(1)
        return b.argmax()*180/b.size(0)

    def forward(self, prev_im: torch.Tensor, curr_im: torch.Tensor):
        lp = self.logPolarNBatch([prev_im,curr_im])
        return self.pcorr_rot_batch_1d(lp)

filterG = getGaussianHpf2d([400,400],22).float()
my_module = MyModule(filterG,(400,400)).cuda()

script_module = torch.jit.script(my_module)
# script_module.save('fmt_scripted.pt')
print(script_module.code)


# import imutils
# boatOriginal = cv2.imread('boat.png',0)
# boatRotated = imutils.rotate(boatOriginal,-25)
# boatOriginal = torch.as_tensor(boatOriginal,dtype=torch.float32,device='cuda')
# boatRotated = torch.as_tensor(boatRotated,dtype=torch.float32,device='cuda')
# trace_module  = torch.jit.trace(my_module,(boatOriginal,boatRotated))
# trace_module.save("fmt_traced.pt")

# torch.onnx.export(my_module,                                # model being run
#                   (torch.randn(400, 400,device='cuda'), torch.randn(400, 400,device='cuda')),    # model input (or a tuple for multiple inputs)
#                   "fmt.onnx",           # where to save the model (can be a file or file-like object)
#                   opset_version=16,
#                   verbose=False,
#                   input_names = ['input'],              # the model's input names
#                   output_names = ['output'])            # the model's output names

# def logPolarNBatch_1d(self, imgs):
#     img = torch.stack(imgs, 0)
#     l_img = len(imgs)
#     imgApo = img*self.apofield + img.masked_select(self.apomask).view(
#         [l_img, -1]).mean(1).view([l_img, 1, 1])*(1-self.apofield)
#     imgDftHpf = (fft.fft_fftshift(
#         torch.fft.fft(torch.fft.fft(imgApo,dim=1),dim=2), [1, 2])*self.filterG).abs()

#     return F.grid_sample(imgDftHpf.unsqueeze(1),
#                             self.polarmap.expand([l_img, -1, -1, -1]),
#                             mode='bicubic',
#                             align_corners=True).squeeze(1)



# def pcorr_rot_batch(self, lp):
#         f = fft.fft_rfft2(lp)
#         scps1 = fft.fft_fftshift(fft.fft_irfft2(
#             (f[0]*f[1].conj())/(f[0].abs()*f[1].abs()+f[1].abs().max()*1e-15)).abs())

#         flat = torch.argmax(scps1)
#         rough0 = torch.div(flat, scps1.size(1), rounding_mode='floor')
#         subarr_index = self.index[0]+flat
#         if subarr_index.lt(0).any() or subarr_index.gt(scps1.numel()).any():
#             # If there is negative index,
#             # Phase correlation failed. So return 0 update
#             return self.mea_return_zero[0]
#         else:
#             # Else calculate from subarr the rotation change
#             subarr = torch.index_select(scps1.view(-1), 0, subarr_index)
#             return (torch.sum(subarr * self.col) / subarr.sum() + rough0 - 2 - f.shape[1] // 2)*180/f.shape[1]
