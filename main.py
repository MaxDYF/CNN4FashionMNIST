import torch
import pandas

# 训练参数设置
batch_size = 64
num_epochs = 100
learning_rate = 0.001

# 超参数设置
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
SENSOR_SIZE = 5
SENSOR_STEP = 1

POOL_SIZE = 2

# 通过超参数计算参数
CONV_LAYER_WIDTH =  (IMAGE_WIDTH - SENSOR_SIZE) // SENSOR_STEP + 1
CONV_LAYER_HEIGHT = (IMAGE_HEIGHT - SENSOR_SIZE) // SENSOR_STEP + 1
DATA_HEIGHT = IMAGE_HEIGHT
DATA_WIDTH = IMAGE_WIDTH
POOL_LAYER_HEIGHT = (CONV_LAYER_HEIGHT // POOL_SIZE)
POOL_LAYER_WIDTH = (CONV_LAYER_WIDTH // POOL_SIZE)
class FCNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(
            POOL_LAYER_WIDTH * POOL_LAYER_WIDTH,
            64
        )
        self.layer2 = torch.nn.Linear(64, 64)
        self.layer3 = torch.nn.Linear(64, 10)
    def forward(self, x):
        x = x.reshape(-1, POOL_LAYER_HEIGHT * POOL_LAYER_WIDTH)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x


class CNNet(torch.nn.Module):
    def __init__(self):
        super(CNNet, self).__init__()
        # 卷积层
        self.conv_layer = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=SENSOR_SIZE,
            stride=SENSOR_STEP
        )
        # 激励层
        self.relu_layer = torch.nn.ReLU()
        # 池化层
        self.max_pool_layer = torch.nn.MaxPool2d(kernel_size=POOL_SIZE)
        self.fc_layer = FCNet()

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.relu_layer(x)
        x = self.max_pool_layer(x)
        x = self.fc_layer(x)
        return x

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

def evaluate(net, data, device):
    n_total = 0
    n_accepted = 0
    with torch.no_grad():
        for image, target in data:
            inputs = torch.unsqueeze(image.to(device), 1)
            results = net.forward(inputs).to(device)
            for i, result in enumerate(results):
                if torch.argmax(result) == target[i]:
                    n_accepted += 1
                n_total += 1
    return n_accepted / n_total

def train(device):
    data = MnistDataset("./fashionmnist/fashion-mnist_train.csv")
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    net = CNNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for image, target in dataloader:
            net.zero_grad()
            inputs = torch.unsqueeze(image.to(device), 1)
            result = net.forward(inputs).to(device)
            loss = torch.nn.functional.nll_loss(result, target.to(device)).to(device)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Accuracy: {evaluate(net, dataloader, device)}")
    return net

def test(net, device):
    data = MnistDataset("./fashionmnist/fashion-mnist_test.csv")
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    print(f"Testing Accuracy: {evaluate(net, dataloader, device)}")

def main():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    net = train(device)
    test(net, device)

if __name__ == "__main__":
    main()

