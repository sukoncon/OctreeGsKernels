import torch 
import torch.nn as nn
import torch.nn.functional as F

import time 

class FusedMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights1, bias1, weights2, bias2):
        # 保存前向传播的中间结果
        
        
        # 第一层 Linear
        z1 = x @ weights1.T + bias1
        a1 = F.relu(z1)
        
        # 第二层 Linear
        z2 = a1 @ weights2.T + bias2
        a2 = F.tanh(z2)
        ctx.save_for_backward(input, z1, a1, z2, weights1, bias1, weights2, bias2)
        
        return a2

    @staticmethod
    def backward(ctx, dC):
        # 获取前向传播的中间结果
        input, z1, a1, z2, weights1, bias1, weights2, bias2 = ctx.saved_tensors
        
        # 反向传播
        # 第二层 Linear 的反向传播
        da2 = dC * (1 - F.tanh(z2) ** 2)  # Tanh 的梯度
        dz2 = da2
        dweights2 = a1.T @ dz2
        dbias2 = torch.sum(dz2, axis=0)
        da1 = dz2 @ weights2
        
        # 第一层 Linear 的反向传播
        dz1 = da1 * (z1 > 0)  # ReLU 的梯度
        dweights1 = input.T @ dz1
        dbias1 = torch.sum(dz1, axis=0)
        dx = dz1 @ weights1
        
        return dx, dweights1.T, dbias1, dweights2.T, dbias2

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
input = torch.randn((M,K0), dtype = torch.float, device =  device, requires_grad = False)

model = nn.Sequential(
          nn.Linear(K0,N0),
          nn.ReLU(True),
          nn.Linear(K1,N1),
          nn.Tanh(),
        ).cuda()

weights = []
bias = []
for name, param in model.named_parameters():
    if "bias" in name:
        bias.append(param)
    elif "weight" in name:
        weights.append(param)
weight0_torch, weight1_torch = weights; weight0_kernel, weight1_kernel = weight0_torch.clone(), weight1_torch.clone()
bias0_torch, bias1_torch = bias; bias0_kernel, bias1_kernel = bias0_torch.clone(), bias1_torch.clone()
weight0_kernel.retain_grad(); bias0_kernel.retain_grad()

# pytorch forward
output = model(input)
target = output.mean()
loss_torch = nn.MSELoss()(output, target)

# kernel forward
output = FusedMLP.apply(input, weight0_kernel, bias0_kernel, weight1_kernel, bias1_kernel)
target = output.mean()
loss_kernel = nn.MSELoss()(output, target)

for i in range(5):

    torch.cuda.synchronize(); time_begin = time.time()
    loss_torch.backward(retain_graph=True)
    torch.cuda.synchronize(); time_end = time.time()
    print(f"torch 2-layer backward used time {round((time_end-time_begin), 4)*1000}ms") # 0.1s+0.3

    torch.cuda.synchronize(); time_begin = time.time()
    loss_kernel.backward(retain_graph=True)
    torch.cuda.synchronize(); time_end = time.time()
    print(f"handwrit 2-layer backward used time {round((time_end-time_begin), 4)*1000}ms") # 0.1s+0.3

    # for name, param in model.named_parameters():
    #     print(f"Layer: {name}")
    #     print(f"Weight Grad:\n{param.grad}")
        # print(f"Bias Grad:\n{param.grad}")
        # print("-" * 50)
    # print(model.state_dict()["0.weight"].grad)

    # torch.cuda.synchronize(); time_begin = time.time()
    # outKernel = FusedMatmul.simple2layer(input, weight0, bias0, "relu", weight1, bias1, "tanh")
    # torch.cuda.synchronize(); time_end = time.time()
    # print(f"simple_gemm used time {round((time_end-time_begin), 4)*1000}ms")
    # print(f"\n")

    close = torch.allclose(weight0_torch.grad.to(torch.float), weight0_kernel.grad.to(torch.float), rtol=1e-05, atol=1e-3, equal_nan=False) 
    print(f"C_simple Accuracy passed? {close}")
    # import pdb; pdb.set_trace()