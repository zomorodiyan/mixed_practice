import torch
import torch.nn
from torch.autograd import Variable

def prep_param_lists(model):
  model_params = [p for p in model.parameters() if p.requires_grad]
  master_params = [p.clone().detach().float() for p in model_params]

  for p in master_params:
      p.requires_grad = True

  return model_params, master_params

def master_params_to_model_params(model_params, master_params):
  for model, master in zip(model_params, master_params):
      model.data.copy_(master.data)

def model_grads_to_master_grads(model_params, master_params):
  for model, master in zip(model_params, master_params):
    if master.grad is None:
      master.grad = Variable(
        master.data.new(*master.data.size()))
    master.grad.data.copy_(model.grad.data)
N, D_in, D_out = 64, 1024, 512
scale_factor = 128.0

x = Variable(torch.randn(N, D_in)).cuda().half()
y = Variable(torch.randn(N, D_out)).cuda().half()

model = torch.nn.Linear(D_in, D_out).cuda().half()
model_params, master_params = prep_param_lists(model)

optimizer = torch.optim.SGD(master_params, lr=1e-3)

for t in range(500):
  y_pred = model(x)

  loss = torch.nn.functional.mse_loss(y_pred.float(), y.float())

  scaled_loss = scale_factor * loss.float()

  model.zero_grad()
  loss.backward()
  model_grads_to_master_grads(model_params, master_params)

  for param in master.params:
    param.grad.data.mul_(1./scale_factor)

  optimizer.step()
  master_params_to_model_params(model_params, master_params)

# BatchNorm involves a reduction -> good idea to use FP32
def BN_convert_float(module):
  if isinstance(module, torch.nn.module.batchnorm._BatchNorm):
    module.float()
  for child in module.children():
    BN_convert_float(child)
  return module

# checkpointing (save/load)
# dynamic loss scaling, start with 2^32, s=s/2 if Nan/Inf. increase after lots
#   of steps without nan/infs
