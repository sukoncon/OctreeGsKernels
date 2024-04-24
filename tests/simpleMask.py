import torch
import CudaKernels
import time

N = 904794
device = torch.device("cuda")
input = torch.randn((N,10,3), device = device).to(torch.int32)
visible_mask = torch.randint(low=0, high=2, size=(N,), device='cuda').bool()
visible_idx = torch.nonzero(visible_mask)
import pdb; pdb.set_trace()
for i in range(10):
    torch.cuda.synchronize(); t0 = time.time()
    ans_torch = input[visible_mask]
    torch.cuda.synchronize(); t1 = time.time()
    print(f"torch used time {(t1-t0)*1000} ms")

    torch.cuda.synchronize(); t0 = time.time()
    ans_kernel = CudaKernels.simpleMask(input, visible_idx)
    torch.cuda.synchronize(); t1 = time.time()
    print(f"kernel used time {(t1-t0)*1000} ms")

diff = (ans_torch - ans_kernel).abs().max()
print(f"max diff {diff}")