from cnn_net import *
from dataset import *
from evaluate import *
import argparse
def test():
    parser = argparse.ArgumentParser(description='CNN for FashionMNIST')
    parser.add_argument('--input', help='Choose Input Training Data.', type=str)
    parser.add_argument('--model', help='Choose Trained Model.', type=str)
    parser.add_argument('--device', help='Choose Device', type=str)
    args = parser.parse_args()
    device = torch.device('cpu')
    if args.device:
        if args.device == 'cuda':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif args.device == 'mps':
            device = torch.device("mps" if torch.mps.is_available() else "cpu")
        elif args.device == 'xpu':
            device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    test_data = MnistDataset(args.input)
    net = CNNet().to(device)
    net.load_state_dict(torch.load(args.model))
    dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    print(f"Testing Accuracy: {evaluate(net, dataloader, device)}")

if __name__ == '__main__':
    test()