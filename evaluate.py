import torch
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