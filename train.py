
import torch
from models.liquid_cell import LiquidNeuron
from models.markov_bp import markov_backprop
from models.loss import mse_loss
from models.optimizer import ManualSGD

input_size = 5
hidden_size = 5
seq_len = 10
batch_size = 1

model = LiquidNeuron(input_size, hidden_size)
optimizer = ManualSGD(model.parameters(), lr=0.01)

for epoch in range(10):
    x_seq = torch.randn(seq_len, input_size)
    target = torch.ones(hidden_size)
    h = torch.zeros(hidden_size)
    states = []

    # Forward pass
    for x in x_seq:
        h = model(h, x)
        states.append(h)

    # Loss
    loss = mse_loss(h, target)
    loss.backward()

    # Manually call optimizer (or use built-in if preferred)
    optimizer.step(model.parameters())
    optimizer.zero_grad()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
