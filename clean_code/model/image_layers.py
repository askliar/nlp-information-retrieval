import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP1(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP1, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
    def forward(self, input_img):
        x = self.fc1(input_img)
        return x


class MLP2(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, input_img):
        x = self.fc1(input_img)
        x = self.fc2(F.relu(x))
        return x