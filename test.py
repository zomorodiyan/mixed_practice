import torch
import torch.nn

bsz, in1, out = 256, 1024, 2048

tensor = torch.randn(bsz, in1).cuda().half()
layer = torch.nn.Linear(in1, out).cuda().half()
layer(tensor)
