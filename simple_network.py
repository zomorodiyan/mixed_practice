import torch
import torch.nn
from torch.autograd import Variable

"""
def prep_param_lists(model):
  '''
  Helper Functions
  '''
  model_params = [p for p in model.parameters() if p.requires_grad]
  master_params = [p.clone().detach().float() for p in model_params]

  for p in master_params:
      p.requires_grad = True

  return model_params, master_params

def master_params_to_model_params(model_params, master_params):
  for model, master in zip(model_params, master_params):
      model.data.copy_(master.data)
      model.data.copy_(master.data)
"""


N, D_in, D_out = 64, 1024, 512

#x = Variable(torch.randn(N, D_in)).cuda().half()
#y = Variable(torch.randn(N, D_out)).cuda().half()
x = Variable(torch.randn(N, D_in)).cuda()
y = Variable(torch.randn(N, D_out)).cuda()

model = torch.nn.Linear(D_in, D_out).cuda()
# model_params, master_params = prep_param_lists(model)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for t in range(500):
  y_pred = model(x)

  loss = torch.nn.functional.mse_loss(y_pred, y)

  # model.zero_grad()
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

# >>> mixed solution >>>
# 1. Forward Pass (FP16 Weights -> FP16 Loss)
# 2. Backprop  (FP16 Loss -> FP16 Gradients)
# 3. Copy (FP16 Gradients -> FP32 Master Gradients)
# 4. Apply (FP32 Master Gradients -> FP32 Master Weights)
# 5. Copy (FP32 Master Weights -> FP16 Weights)
# <<< mixed solution <<<

