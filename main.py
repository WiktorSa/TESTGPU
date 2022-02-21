import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time


class ControllerDataset(Dataset):
    def __init__(self, x_A: np.ndarray, y_A: np.ndarray):
        self.x_A = x_A
        self.y_A = y_A

    def __len__(self) -> int:
        return len(self.x_A)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.Tensor(self.x_A[idx]), torch.Tensor(self.y_A[idx])


class ControllerModel(nn.Module):
    def __init__(self):
        super(ControllerModel, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(10, 1024),
            nn.Linear(1024, 2048),
            nn.Linear(2048, 9092),
            nn.Linear(9092, 20)
        )

    def forward(self, data):
        return self.linear(data)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    x = np.empty(shape=(1000, 10))
    y = np.empty(shape=(1000, 20))
    dataset = ControllerDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32)

    model = ControllerModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    or_time = time.time()
    for epoch in range(20):
        dev_time = time.time()
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        print(time.time() - dev_time)

    print(time.time() - or_time)
