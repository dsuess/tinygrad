import time
import torch

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

for dtype in [torch.float16, torch.float32]:
  for N in [4096]:
    FLOPS = N*N*N*2

    b = torch.rand((N,N), dtype=dtype).cuda()
    c = torch.rand((N,N), dtype=dtype).cuda()

    def torch_prog(b, c):
      st = time.perf_counter()
      a = b@c
      torch.cuda.synchronize()
      return time.perf_counter() - st
    tm = min([torch_prog(b, c) for _ in range(100)])
    print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS {N:4d}x{N:4d}x{N:4d} matmul in {dtype}")
