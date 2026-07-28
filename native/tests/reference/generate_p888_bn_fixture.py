import json
from pathlib import Path

import torch
from torch import nn


torch.set_default_dtype(torch.float32)
STATE_LEN, STATE_PAD, HD1, HD2, BLOCKS, OUTPUT = 2, 4, 3, 2, 1, 1


class FixtureNet(nn.Module):
    def __init__(self, flat):
        super().__init__()
        cursor = 0

        def take(count):
            nonlocal cursor
            result = flat[cursor : cursor + count]
            cursor += count
            return result

        self.table = nn.Parameter(take(STATE_LEN * STATE_PAD * HD1).reshape(STATE_LEN * STATE_PAD, HD1))
        self.input_bias = nn.Parameter(take(HD1))
        self.input_bn = nn.BatchNorm1d(HD1)
        self.hidden = nn.Linear(HD1, HD2)
        self.hidden.weight.data.copy_(take(HD1 * HD2).reshape(HD1, HD2).T)
        self.hidden.bias.data.copy_(take(HD2))
        self.hidden_bn = nn.BatchNorm1d(HD2)
        self.fc1 = nn.Linear(HD2, HD2)
        self.fc1.weight.data.copy_(take(HD2 * HD2).reshape(HD2, HD2).T)
        self.fc1.bias.data.copy_(take(HD2))
        self.fc1_bn = nn.BatchNorm1d(HD2)
        self.fc2 = nn.Linear(HD2, HD2)
        self.fc2.weight.data.copy_(take(HD2 * HD2).reshape(HD2, HD2).T)
        self.fc2.bias.data.copy_(take(HD2))
        self.fc2_bn = nn.BatchNorm1d(HD2)
        self.output = nn.Linear(HD2, OUTPUT)
        self.output.weight.data.copy_(take(HD2 * OUTPUT).reshape(HD2, OUTPUT).T)
        self.output.bias.data.copy_(take(OUTPUT))
        assert cursor == flat.numel()

    def forward(self, states):
        rows = torch.arange(STATE_LEN)
        indices = rows[None, :] * STATE_PAD + states
        x = self.table[indices].sum(1) + self.input_bias
        x = torch.relu(self.input_bn(x))
        x = torch.relu(self.hidden_bn(self.hidden(x)))
        skip = x
        x = torch.relu(self.fc1_bn(self.fc1(x)))
        x = torch.relu(skip + self.fc2_bn(self.fc2(x)))
        return self.output(x)


param_count = STATE_LEN * STATE_PAD * HD1 + HD1 + HD1 * HD2 + HD2 + 2 * (HD2 * HD2 + HD2) + HD2 * OUTPUT + OUTPUT
weights = torch.tensor([0.03 * ((i % 9) - 4) for i in range(param_count)])
states = torch.tensor([[i, (i + 1) % STATE_PAD] for i in range(4)], dtype=torch.long)
model = FixtureNet(weights)
model.train()
train_prediction = model(states)
labels = torch.tensor([[1.0], [3.0], [-1.0], [2.0]])
loss = torch.mean((train_prediction - labels) ** 2)
loss.backward()
train_output = train_prediction.detach().flatten()
weight_grad = torch.cat([
    model.table.grad.flatten(), model.input_bias.grad.flatten(),
    model.hidden.weight.grad.T.flatten(), model.hidden.bias.grad.flatten(),
    model.fc1.weight.grad.T.flatten(), model.fc1.bias.grad.flatten(),
    model.fc2.weight.grad.T.flatten(), model.fc2.bias.grad.flatten(),
    model.output.weight.grad.T.flatten(), model.output.bias.grad.flatten(),
])
batch_norms = (model.input_bn, model.hidden_bn, model.fc1_bn, model.fc2_bn)
affine_grad = torch.cat([
    *(bn.weight.grad.flatten() for bn in batch_norms),
    *(bn.bias.grad.flatten() for bn in batch_norms),
])
running = []
for bn in batch_norms:
    running.extend(bn.running_mean.tolist())
for bn in batch_norms:
    running.extend(bn.running_var.tolist())
model.eval()
eval_output = model(states).detach().flatten()

payload = {
    "shape": [STATE_LEN, STATE_PAD, HD1, HD2, BLOCKS, OUTPUT],
    "weights": weights.tolist(),
    "states": states.tolist(),
    "train_output": train_output.tolist(),
    "labels": labels.flatten().tolist(),
    "loss": loss.item(),
    "weight_grad": weight_grad.tolist(),
    "affine_grad": affine_grad.tolist(),
    "running": running,
    "eval_output": eval_output.tolist(),
}
destination = Path(__file__).parents[1] / "fixtures" / "p888_bn_contract.json"
destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(destination)
