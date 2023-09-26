import argparse
import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as data_utils

tensor_transform = transforms.ToTensor()

data = datasets.MNIST(root="./data", train=True, download=True, transform=tensor_transform)
loader = data_utils.DataLoader(dataset=data, batch_size=32, shuffle=True)


def plot(image, num_images):
    if num_images == 1:
        fig = plt.figure
        plt.imshow(image, cmap='gray')
        plt.show()
    else:
        for i in range(num_images):
            fig = plt.figure
            plt.imshow(image[i], cmap='gray')
            plt.show()


class AutoEncoder(torch.nn.Module):
    def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
        super(AutoEncoder, self).__init__()
        N2 = int(N_input / 2)
        self.type = 'MLP4'
        self.input_shape = (1, N_input)

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(N_input, N2),
            torch.nn.ReLU(),
            torch.nn.Linear(N2, N_bottleneck),
            torch.nn.ReLU(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(N_bottleneck, N2),
            torch.nn.ReLU(),
            torch.nn.Linear(N2, N_output),
            torch.nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train(n_epochs, optimizer, model, loss_fn, train_loader, schedule):
    print('training...')
    model.train()
    outputs = []
    losses = []
    loss_train = []
    for epoch in range(n_epochs):
        for (image, _) in train_loader:
            image = image.reshape(-1, 28 * 28)

            reconstructed = model(image)
            loss = loss_fn(reconstructed, image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        loss_train.append(np.mean(losses))
        outputs.append((n_epochs, image, reconstructed))
        print('epoch [{}/{}], loss: {:.4f}'.format(epoch + 1, n_epochs, loss_train[-1]))

        losses = []

    plt.plot(loss_train)
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Curve')
    plt.show()
    plt.savefig()


if __name__ == "__main__":
    model = AutoEncoder()

    parser = argparse.ArgumentParser(description="Train an AutoEncoder model")
    parser.add_argument("-z", "--bottleneck-size", type=int, default=8, help="Bottleneck size")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("-s", "--savepath", type=str, default="MLP.8.pth",
                        help="Path to save the model")
    parser.add_argument("-p", "--plot-path", type=str, default="loss_curve.png",
                        help="Path to save the loss curve plot")
    args = parser.parse_args()

    optimizer = torch.optim.Adam(model.parameters())
    loss_function = torch.nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    train(args.epochs, optimizer, model, loss_function, loader, scheduler)

    torch.save(model, '/Users/markbenhamu/PycharmProjects/CMPE474Lab1/' + args.savepath)
