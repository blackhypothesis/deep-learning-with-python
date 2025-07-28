import torch
import numpy as np
import matplotlib.pyplot as plt

num_samples_per_class = 1000

negative_samples = np.random.multivariate_normal(
    mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
)

inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)

targets = np.vstack(
    (
        np.zeros((num_samples_per_class, 1), dtype="float32"),
        np.ones((num_samples_per_class, 1), dtype="float32"),
    )
)

plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()

input_dim = 2
output_dim = 1

# W = torch.rand(input_dim, output_dim, requires_grad=True)
# b = torch.rand(output_dim, requires_grad=True)

# def model(inputs, W, b):
#     return torch.matmul(inputs, W) + b

def mean_squared_error(targets, predictions):
    per_sample_losses = torch.square(targets - predictions)
    return torch.mean(per_sample_losses)

learning_rate = 0.1

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W = torch.nn.Parameter(torch.rand(input_dim, output_dim))
        self.b = torch.nn.Parameter(torch.zeros(output_dim))

    def forward(self, inputs):
        return torch.matmul(inputs, self.W) + self.b

model = LinearModel()
torch_inputs = torch.tensor(inputs)
torch_targets = torch.tensor(targets)
output = model(torch_inputs)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def training_step(inputs, targets):
    predictions = model(inputs)
    loss = mean_squared_error(targets, predictions)
    loss.backward()
    optimizer.step()
    model.zero_grad()
    return loss

for step in range(40):
    loss = training_step(torch_inputs, torch_targets)
    print(f"Loss at step {step}: {loss:.4f}")

predictions = model(torch_inputs)
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)

W = model.W.detach().numpy()
b = model.b.detach().numpy()

x = np.linspace(-1, 4, 100)
y = -W[0] / W[1] * x + (0.5 - b) / W[1]
plt.plot(x, y, "-r")

plt.show()