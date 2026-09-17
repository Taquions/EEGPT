"""Launch real kernels and check the results, in fp32 and fp16.

`torch.cuda.is_available()` returning True says nothing about whether kernels
actually run on this architecture, so this compares a GPU matmul against the CPU
result and exercises a conv1d backward pass.
"""
import time

import torch

d = torch.device("cuda")

a = torch.randn(4096, 4096, device=d)
b = torch.randn(4096, 4096, device=d)

torch.cuda.synchronize()
t0 = time.time()
c = a @ b
torch.cuda.synchronize()
print("matmul 4096^3 ok, %.3f s" % (time.time() - t0))

ref = a.cpu() @ b.cpu()
err = (c.cpu() - ref).abs().max().item()
rel = err / ref.abs().max().item()
print("max abs error vs cpu: %.4f (relative %.2e)" % (err, rel))
assert rel < 1e-3, "GPU matmul disagrees with CPU -- kernels are not running correctly"

ah, bh = a.half(), b.half()
torch.cuda.synchronize()
t0 = time.time()
_ = ah @ bh
torch.cuda.synchronize()
print("fp16 matmul ok, %.3f s" % (time.time() - t0))

conv = torch.nn.Conv1d(58, 58, 3, padding=1).to(d)
x = torch.randn(16, 58, 1024, device=d)
y = conv(x)
y.sum().backward()
print("conv1d fwd+bwd ok, out:", tuple(y.shape))
