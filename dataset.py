import pandas
import torch
def load_data(src):
    dataset = pandas.read_csv(src)
    # 将一维张量转化为图片形式的二维张量
    x = torch.tensor(dataset.iloc[1:, 1:].values, dtype=torch.float).reshape(-1, 28, 28)
    y = torch.tensor(dataset.iloc[1:, 0].values, dtype=torch.int64)
    return x, y

class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, src):
        super().__init__()
        self.images, self.targets = load_data(src)
    def __getitem__(self, index):
        return self.images[index], self.targets[index]
    def __len__(self):
        return len(self.images)