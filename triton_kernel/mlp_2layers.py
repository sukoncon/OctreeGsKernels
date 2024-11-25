import torch
import triton
from triton import language as tl
# from actual_base_gptq_4 import triton_matmul4
import time
if triton.__version__ >= "3.0.0":
    from triton.language.extra.cuda.libdevice import fast_expf as tl_exp
    from triton.language.extra.cuda.libdevice import fast_logf as tl_log
    from triton.language.extra.cuda.libdevice import tanh as tl_tanh

else:
    from triton.language.math import fast_expf as tl_exp
    from triton.language.math import fast_logf as tl_log
    from triton.language.math import tanh as tl_tanh


@triton.jit()
def tl_activation(x, activation):
    if activation == 1: # relu
        return tl.where(x >= 0, x, 0)
    # if activation == 2: # Leaky ReLU
    #     return tl.where(x >= 0, x, 0)
    if activation == 3: # sigmoid
        return 1 / (1 + tl_exp(-x))
    if activation == 4: # Softplus
        return tl_log(1 + tl_exp(x))
    if activation == 5: # Tanh
        return tl_tanh(x)
    else:
        return x

@triton.autotune(
    configs = [
        triton.Config({'block_m': BM, 'block_k': BK}, num_stages=s, num_warps=w) \
        for BM in [32] \
        for BK in [32] \
        for s in [4] \
        for w in [4] \
    ],
    key=['n0', 'n1'],
)

@triton.jit()
def mlp_2layer_fwd_kernel(in_ptr, 
            w0_ptr, b0_ptr, a0: tl.constexpr,
            w1_ptr, b1_ptr, a1: tl.constexpr,
            out_ptr,
            m, n0, k0, n0_pad: tl.constexpr, k0_pad: tl.constexpr,
            n1, k1, n1_pad: tl.constexpr, k1_pad: tl.constexpr,
            block_m: tl.constexpr, block_k: tl.constexpr):

    pid_m = tl.program_id(0)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_k = tl.arange(0, block_k)

    input_ptrs = in_ptr + offs_m[:, None] * k0 + offs_k[None, :]
    w0_ptrs = w0_ptr + offs_k[:, None] + tl.arange(0, n0_pad)[None, :] * k0
    
    # a block should deal with a set of rows
    out0 = tl.zeros((block_m, n0_pad), dtype = tl.float32)

    # the first layer
    for k in range(0, tl.cdiv(k0, block_k)):
        input = tl.load(input_ptrs, mask = (offs_k[None, :] < k0) & (offs_m[:, None] < m), other = 0.0)
        weight = tl.load(w0_ptrs, mask = offs_k[:, None] < k0, other = 0.0)
        # tl.device_print("before out0", out0)
        out0 += tl.dot(input, weight, out_dtype = tl.float32)
        # tl.device_print("after out0", out0)
        input_ptrs += block_k
        w0_ptrs += block_k
        offs_k += block_k
    
    # tl.debug_barrier()
    b0_ptrs = b0_ptr + tl.arange(0, n0_pad)[None, :] 
    b0 = tl.load(b0_ptrs)
    out0 = b0 + out0
    out0 = tl_activation(out0, a0)

    # the second layer
    w1_ptrs = w1_ptr + tl.arange(0, k1_pad)[:, None] + tl.arange(0, n1_pad)[None, :] * k1
    weight = tl.load(w1_ptrs)
    out1 = tl.dot(out0, weight, out_dtype = tl.float32)

    b1_ptrs = b1_ptr + tl.arange(0, n1_pad)[None, :]
    b1 = tl.load(b1_ptrs, mask = tl.arange(0, n1_pad)[None, :] < n1, other = 0.0)
    out1 += b1
    out1 = tl_activation(out1, a1)
    
    out_ptrs = out_ptr + offs_m[:, None] * n1 + tl.arange(0, n1_pad)[None, :]
    tl.store(out_ptrs, out1, mask = (tl.arange(0, n1_pad)[None, :] < n1) & (offs_m[:, None] < m))
    


def fused2layer(input, 
            w0, b0, a0,
            w1, b1, a1,
            output = None):

    m, k0 = input.shape
    n0, _ = w0.shape
    n1, k1 = w1.shape

    n0_pad = max(triton.next_power_of_2(n0), 16)
    n1_pad = max(triton.next_power_of_2(n1), 16)
    k0_pad = max(triton.next_power_of_2(k0), 16)
    k1_pad = max(triton.next_power_of_2(k1), 16)

    assert n0_pad in [32, 64, 128], "n0 should in [32, 64, 128]"
    assert n1_pad in [32, 64, 128], "n1 should in [32, 64, 128]"

    if output is None:
        # print("Creat c from scratch")
        output = torch.empty((m, n1), device = input.device, dtype = input.dtype)

    grid = lambda META: (triton.cdiv(m, META["block_m"]), 1, 1)
    mlp_2layer_fwd_kernel[grid](input, 
            w0, b0, a0,
            w1, b1, a1,
            output,
            m, n0, k0, n0_pad, k0_pad,
            n1, k1, n1_pad, k1_pad)
    print(mlp_2layer_fwd_kernel.best_config)
    return output

class FusedMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, 
                w0, b0, a0,
                w1, b1, a1):
        ctx.save_for_backward(input)
        triton_out = fused2layer(input, 
                w0, b0, a0,
                w1, b1, a1,
                output = None)
        return triton_out

if __name__ == "__main__":
    import torch.nn as nn


    device = "cuda:0"
    torch.cuda.set_device(device)
    m = 124308
    input = torch.randn((m, 35), dtype = torch.float32, device = device)

    mlp = nn.Sequential(
            nn.Linear(in_features=35, out_features=32),
            nn.Softplus(),
            nn.Linear(in_features=32, out_features=70),
            nn.Tanh(),
        ).cuda()
    
    keys = mlp.state_dict().keys()
    weight_keys = list(filter(lambda x: 'weight' in x, keys))
    bias_keys = list(filter(lambda x: 'bias' in x, keys))

    assert len(weight_keys) == 2, "Only support when nn.Linear layers of mlp == 2"

    w0 = mlp.state_dict()[weight_keys[0]]
    w1 = mlp.state_dict()[weight_keys[1]]

    b0 = mlp.state_dict()[bias_keys[0]]
    b1 = mlp.state_dict()[bias_keys[1]]
    
    def act_fn(activation):
        if isinstance(activation, torch.nn.ReLU):
            return 1 # "ReLU"
        # if isinstance(activation, torch.nn.LeakyReLU):
        #     return 2 # "Leaky ReLU"
        if isinstance(activation, torch.nn.Sigmoid):
            return 3 # "Sigmoid"
        if isinstance(activation, torch.nn.Softplus):
            return 4 # "Softplus"
        if isinstance(activation, torch.nn.Tanh):
            return 5 #"Tanh"
        if isinstance(activation, type(None)):
            return 0 # "None"
        assert 0, "only support when activation = relu or sigmoid"

    if int(weight_keys[1][0]) - int(weight_keys[0][0]) == 2:
        a0 = act_fn(mlp[int(weight_keys[0][0]) + 1])
    else:
        a0 = 0

    if (len(mlp) - 1) == (int(weight_keys[1][0]) + 1):
        a1 = act_fn(mlp[int(weight_keys[1][0]) + 1])
    else:
        a1 = 0

    for i in range(5):
        torch.cuda.synchronize(); start = time.time()
        torch_out = mlp(input)
        torch.cuda.synchronize(); end = time.time()
        print(f"torch_out used time: {(end-start)*1000} ms")

        torch.cuda.synchronize(); start = time.time()
        triton_out = FusedMLP.apply(input, 
                w0, b0, a0,
                w1, b1, a1)
        torch.cuda.synchronize(); end = time.time()
        print(f"triton_out used time: {(end-start)*1000} ms \n")

    print("diff max: ", (torch_out-triton_out).abs().max())
    import pdb; pdb.set_trace()
