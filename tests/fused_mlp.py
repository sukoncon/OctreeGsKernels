import torch 
import torch.nn as nn

import time 
import FusedMatmul
print(f"FusedMatmul {FusedMatmul.__file__}")

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
N0 = 32; N1 = 70; K1 = N0
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
input = torch.randn((M,K0), dtype = torch.float, device =  device)

model = nn.Sequential(
          nn.Linear(K0,N0),
          nn.ReLU(True),
          nn.Linear(K1,N1),
          nn.Tanh(),
        ).cuda()

# model.state_dict()["0.bias"] *= 0
# model.state_dict()["2.bias"] *= 0
# model.state_dict()["0.weight"] *= 0
# model.state_dict()["0.weight"] += 1
# model.state_dict()["2.weight"] *=0 
# model.state_dict()["2.weight"] += 1

weight0 = model.state_dict()["0.weight"]
bias0 = model.state_dict()["0.bias"]
weight1 = model.state_dict()["2.weight"]
bias1 = model.state_dict()["2.bias"]

print(model[0](input))


for i in range(5):

    torch.cuda.synchronize(); time_begin = time.time()
    C_torch = model(input)
    torch.cuda.synchronize(); time_end = time.time()
    print(f"torch single layer used time {round((time_end-time_begin), 4)*1000}ms") # 0.1s+0.3

    torch.cuda.synchronize(); time_begin = time.time()
    outKernel = FusedMatmul.simple2layer(input, weight0, bias0, "relu", weight1, bias1, "tanh")
    torch.cuda.synchronize(); time_end = time.time()
    print(f"simple_gemm used time {round((time_end-time_begin), 4)*1000}ms")
    print(f"\n")

close = torch.allclose(C_torch.to(torch.float), outKernel, rtol=1e-05, atol=1e-3, equal_nan=False) 
print(f"C_simple Accuracy passed? {close}")
# import pdb; pdb.set_trace()