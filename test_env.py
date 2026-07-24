# python交互界面
import torch
print(torch.backends.mps.is_available()) # True = GPU可用
print(torch.backends.mps.is_built())