import torch 
import torch.nn as nn

import time 
import MyMatmul
print(f"MyMatmul {MyMatmul.__file__}")

device = torch.device("cuda")
# cov_mlp = torch.jit.load("cov_mlp.pt").to(device)
'''
RecursiveScriptModule(
  original_name=Sequential
  (0): RecursiveScriptModule(original_name=Linear)
  (1): RecursiveScriptModule(original_name=ReLU)
  (2): RecursiveScriptModule(original_name=Linear)
) 
'''

# input = torch.load("cat_local_view_wodist.pt").to(device) # [M, 35] (35,1) [427154, 35]
M = 427154; K0 = 35
N0 = 17; N1 = 70; K1 = N0
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
input = torch.randn((M,K0), dtype = torch.float, device =  device)

model = nn.Sequential(
          nn.Linear(K0,N0),
          nn.Linear(K1,N1),
        ).cuda()
# model.state_dict()["0.bias"] *= 0
# model.state_dict()["1.bias"] *= 0
# model.state_dict()["0.weight"] *= 0
# model.state_dict()["0.weight"] += 1
# model.state_dict()["1.weight"] *=0 
# model.state_dict()["1.weight"] += 1
weight0 = model.state_dict()["0.weight"]
bias0 = model.state_dict()["0.bias"]
weight1 = model.state_dict()["1.weight"]
bias1 = model.state_dict()["1.bias"]

print(model[0](input))
# import pdb; pdb.set_trace()
# cov_mlp.state_dict()['0.bias'].shape #32
# cov_mlp.state_dict()['0.weight'].shape # [32, 35] N K
# cov_mlp.state_dict()['0.weight'].stride() # [35, 1]
# cov_mlp.state_dict()['2.bias'].shape #70
# cov_mlp.state_dict()['2.weight'].shape # [70, 32]
# cov_mlp.state_dict()['2.weight'].stride() # (32, 1)

for i in range(5):
    # torch.cuda.synchronize(); t0 = time.time()
    # ans_torch = cov_mlp(input) # 0.7 ms
    # torch.cuda.synchronize(); t1 = time.time()
    # print(f"\ntorch used time {(t1-t0)*1000} ms")

    torch.cuda.synchronize(); time_begin = time.time()
    C_torch = model(input)
    torch.cuda.synchronize(); time_end = time.time()
    print(f"torch single layer used time {round((time_end-time_begin), 4)*1000}ms") # 0.1s+0.3

    import MyMatmul
    torch.cuda.synchronize(); time_begin = time.time()
    outKernel = MyMatmul.simple_fused_gemm(input, weight0, bias0, "None", weight1, bias1, "None")
    torch.cuda.synchronize(); time_end = time.time()
    print(f"simple_gemm used time {round((time_end-time_begin), 4)*1000}ms")
    print(f"\n")

close = torch.allclose(C_torch.to(torch.float), outKernel, rtol=1e-03, atol=1, equal_nan=False) 
print(f"C_simple Accuracy passed? {close}")
import pdb; pdb.set_trace()
# if input.is_contiguous:
#   M0 = input.size(0)
#   K0 = input.size(1)
#   lda0 = input.stride(0)
# else:
#   error info ("not implemented for discontigous tensor type")
# if input.is_contiguous:
#   M0 = A.size(0)
#   K0 = A.size(1)
#   lda0 = A.stride(0)
(outKernel-C_torch)[:, 0].max()

(outKernel-C_torch)[:, 0].argmax()