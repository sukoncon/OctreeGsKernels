import torch
import triton
from triton import language as tl
# from actual_base_gptq_4 import triton_matmul4
import time
if triton.__version__ >= "3.0.0":
    from triton.language.extra.cuda.libdevice import fast_expf as tl_exp
    from triton.language.extra.cuda.libdevice import fast_logf as tl_log
    from triton.language.extra.cuda.libdevice import fast_tanf as tl_tan
    from triton.language.extra.cuda.libdevice import fast_powf as tl_pow
else:
    from triton.language.math import fast_expf as tl_exp
    from triton.language.math import fast_logf as tl_log


@triton.autotune(
    configs = [
        triton.Config({'block_m': BM, 'block_k': BK}, num_stages=s, num_warps=w) \
        for BM in [128] \
        for BK in [32] \
        for s in [1] \
        for w in [4] \
    ],
    key=['n0', 'n1', 'n2'],
)

@triton.jit()
def mlp_3layer_fwd_kernel(in_ptr, 
            w0_ptr, b0_ptr, a0,
            w1_ptr, b1_ptr, a1,
            w2_ptr, b2_ptr, a2,
            out_ptr,
            m, n0, k0, n0_pad: tl.constexpr, k0_pad: tl.constexpr,
            n1, k1, n1_pad: tl.constexpr, k1_pad: tl.constexpr,
            n2, k2, n2_pad: tl.constexpr, k2_pad: tl.constexpr,
            block_m: tl.constexpr, block_k: tl.constexpr):

    pid_m = tl.program_id(0)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_k = tl.arange(0, block_k)

    input_ptrs = in_ptr + offs_m[:, None] * k0 + offs_k[None, :]
    w0_ptrs = w0_ptr + offs_k[:, None] + tl.arange(0, n0_pad)[None, :] * k0
    
    # a block should deal with a set of rows
    out0 = tl.zeros((block_m, n0_pad), dtype = tl.float32)
    # out1 = tl.zeros((block_m, n1_pad), dtype = tl.float32)
    # out2 = tl.zeros((block_m, n2_pad), dtype = tl.float32)

    # the first layer
    for k in range(0, tl.cdiv(k0, block_k)):
        input = tl.load(input_ptrs, mask = offs_k[None, :] < k0, other = 0.0).to(tl.float32)
        weight = tl.load(w0_ptrs, mask = offs_k[:, None] < k0, other = 0.0).to(tl.float32)
        # tl.device_print("before out0", out0)
        out0 += tl.dot(input, weight, out_dtype = tl.float32)
        # tl.device_print("after out0", out0)
        input_ptrs += block_k
        w0_ptrs += block_k
        offs_k += block_k
    
    # tl.debug_barrier()
    b0_ptrs = b0_ptr + tl.arange(0, n0_pad)[None, :] 
    b0 = tl.load(b0_ptrs).to(tl.float32)
    out0 = b0 + out0
    
    

    # the second layer
    w1_ptrs = w1_ptr + tl.arange(0, k1_pad)[:, None] + tl.arange(0, n1_pad)[None, :] * k1
    weight = tl.load(w1_ptrs).to(tl.float32)
    out1 = tl.dot(out0, weight, out_dtype = tl.float32)

    b1_ptrs = b1_ptr + tl.arange(0, n1_pad)[None, :]
    b1 = tl.load(b1_ptrs, mask = tl.arange(0, n1_pad)[None, :] < n1, other = 0.0).to(tl.float32)
    out1 += b1
    
    

    # the third layer
    w2_ptrs = w2_ptr + tl.arange(0, k2_pad)[:, None] + tl.arange(0, n2_pad)[None, :] * k2
    weight = tl.load(w2_ptrs).to(tl.float32)
    out2 = tl.dot(out1, weight, out_dtype = tl.float32)

    b2_ptrs = b2_ptr + tl.arange(0, n2_pad)[None, :]
    b2 = tl.load(b2_ptrs, mask = tl.arange(0, n2_pad)[None, :] < n2, other = 0.0).to(tl.float32)
    out2 += b2
    # tl.device_print("out2", out2)

    out_ptrs = out_ptr + offs_m[:, None] * n2 + tl.arange(0, n2_pad)[None, :]
    tl.store(out_ptrs, out2, mask = tl.arange(0, n2_pad)[None, :] < n2)
    


def fused3layer(input, 
            w0, b0, a0,
            w1, b1, a1,
            w2, b2, a2, output = None):

    m, k0 = input.shape
    n0, _ = w0.shape
    n1, k1 = w1.shape
    n2, k2 = w2.shape

    n0_pad = max(triton.next_power_of_2(n0), 16)
    n1_pad = max(triton.next_power_of_2(n1), 16)
    n2_pad = max(triton.next_power_of_2(n2), 16)
    k0_pad = max(triton.next_power_of_2(k0), 16)
    k1_pad = max(triton.next_power_of_2(k1), 16)
    k2_pad = max(triton.next_power_of_2(k2), 16)

    assert n0 in [32, 64, 128], "n0 should in [32, 64, 128]"
    assert n1 in [32, 64, 128], "n1 should in [32, 64, 128]"

    if output is None:
        # print("Creat c from scratch")
        output = torch.empty((m, n2), device = input.device, dtype = input.dtype)
    # import pdb; pdb.set_trace()

    grid = lambda META: (triton.cdiv(m, META["block_m"]), 1, 1)
    mlp_3layer_fwd_kernel[grid](input, 
            w0, b0, a0,
            w1, b1, a1,
            w2, b2, a2,
            output,
            m, n0, k0, n0_pad, k0_pad,
            n1, k1, n1_pad, k1_pad,
            n2, k2, n2_pad, k2_pad)
    return output

if __name__ == "__main__":
    import torch.nn as nn


    device = "cuda:0"
    torch.cuda.set_device(device)
    m = 128*100
    input = torch.randn((m, 150), dtype = torch.float32, device = device)

    mlp = nn.Sequential(
            nn.Linear(in_features=150, out_features=64),
            # nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=64),
            # nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=3),
            # nn.Sigmoid(),
        ).cuda()
    
    keys = mlp.state_dict().keys()
    weight_keys = list(filter(lambda x: 'weight' in x, keys))
    bias_keys = list(filter(lambda x: 'bias' in x, keys))

    assert len(weight_keys) == 3, "Only support when layers of mlp == 3"

    weights = []
    bias = []

    def judge_activation_layer(layer):
        """
        Given a PyTorch layer object, determine if it uses ReLU or Sigmoid activation function.

        Args:
            layer (nn.Module): The PyTorch module representing the layer you want to check.

        Returns:
            int: Returns 1 for ReLU layers, 2 for sigmoid layers, else raises an assertion error.
        """
        
        # Check if the layer is an instance of either ReLU or Sigmoid class from `nn`
        if isinstance(layer, nn.ReLU):
            return 1
        elif isinstance(layer, nn.Sigmoid):
            return 2
    
    w0 = mlp.state_dict()[weight_keys[0]]
    w1 = mlp.state_dict()[weight_keys[1]]
    w2 = mlp.state_dict()[weight_keys[2]]

    b0 = mlp.state_dict()[bias_keys[0]]
    b1 = mlp.state_dict()[bias_keys[1]]
    b2 = mlp.state_dict()[bias_keys[2]]
    
    # w0 *= 0; w0 += 0;
    # w1 *= 0; w1 += 0;
    # w2 *= 0; w2 += 0;
    # b0 *= 0; b1 *= 0; b2 *= 0; 
    # b0 += 2; b1 += 1; b2 += 1;     w2 *= 0; w2 += 1;


    a0 = 0
    a1 = 0
    a2 = 0
    for i in range(10):
        torch.cuda.synchronize(); start = time.time()
        torch_out = mlp(input)
        torch.cuda.synchronize(); end = time.time()
        print(f"torch_out used time: {(end-start)*1000} ms")

        torch.cuda.synchronize(); start = time.time()
        triton_out = fused3layer(input, 
                w0, b0, a0,
                w1, b1, a1,
                w2, b2, a2, output = None)
        torch.cuda.synchronize(); end = time.time()
        print(f"triton_out used time: {(end-start)*1000} ms \n")

    print("diff max: ", (torch_out-triton_out).abs().max())
