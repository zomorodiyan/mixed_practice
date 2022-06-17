import torch
import torch.nn
from torch.autograd import Variable


N, D_in, D_out = 64, 1024, 512

x = Variable(torch.randn(N, D_in)).cuda()
y = Variable(torch.randn(N, D_out)).cuda()

model = torch.nn.Linear(D_in, D_out).cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for t in range(500):
  y_pred = model(x)

  loss = torch.nn.functional.mse_loss(y_pred, y)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
