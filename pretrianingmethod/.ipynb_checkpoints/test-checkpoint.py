import torch

# 假设你有一个张量 tensor


tensor1 = torch.tensor([[0, 1, 2, 0, 0, 4, 0],
                      [0, 1, 2, 0, 0, 4, 0],
                      [0, 1, 2, 0, 0, 4, 0]])
tensor2 = torch.tensor([[0, 1, 2, 3, 0, 4, 5],
                      [0, 1, 2, 3, 0, 4, 5],
                      [0, 1, 2, 3, 0, 4, 5]])

# 将特定值（比如2和3）置为1
specific_values = [2,3]
tensor2[torch.where(tensor1==2)] = 0.0
# a=torch.where(tensor == specific_values,torch.tensor(1),tensor)
a =tensor2
print(a)