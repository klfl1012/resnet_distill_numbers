import torch
import torch.nn as nn 


class StudentCNN(nn.Module):

    def __init__(self, num_filters1, num_filters2, kernel_size1, kernel_size2, padding1, padding2, padding3, hidden_units):
        super(StudentCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, num_filters1, kernel_size=kernel_size1, padding=padding1)
        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size=kernel_size2, padding=padding2)
        self.conv3 = nn.Conv2d(num_filters2, num_filters2 * 2, kernel_size=3, padding=padding3)
        self.pool = nn.MaxPool2d(2, 2)

        self._to_linear = self._compute_flattened_size(112, 112)

        self.fc1 = nn.Linear(self._to_linear, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 10)

    def _compute_flattened_size(self, h, w):
        with torch.no_grad():
            x = torch.randn(1, 1, h, w)
            x = self.pool(nn.functional.relu(self.conv1(x)))
            x = self.pool(nn.functional.relu(self.conv2(x)))
            x = self.pool(nn.functional.relu(self.conv3(x)))
            x = x.numel() // x.size(0)
            return x

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x