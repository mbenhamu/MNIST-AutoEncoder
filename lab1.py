import argparse

from train import AutoEncoder
from torchvision import transforms

tensor_transform = transforms.ToTensor()
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def image_comp(index, model, data):
    image, _ = data[index]
    image = image.reshape(-1, 784)

    fed = model(image)
    img = image.detach().numpy()
    output = fed.detach().numpy()

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output.reshape(28, 28), cmap='gray')
    plt.title('Reconstructed Image')
    plt.show()


def reconstruct_and_plot_image(index, model, data):
    # Get the specified image from the dataset
    image, _ = data[index]
    image = image.reshape(-1, 28 * 28)
    fed = model(image)
    noisy_img = image + 0.3 * torch.rand(image.shape)
    # Pass the image through the model
    reconstructed_noise_image = model(noisy_img)

    # Convert tensors to numpy arrays
    image = image.detach().numpy()
    out2 = reconstructed_noise_image.detach().numpy()

    # Plot the original and reconstructed images
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_img.reshape(28, 28), cmap='gray')
    plt.title('Noisy Image')

    plt.subplot(1, 3, 3)
    plt.imshow(out2.reshape(28, 28), cmap='gray')
    plt.title('Reconstructed Noisy')
    plt.show()


def interpolate_and_plot(model, index1, index2, data, n_steps=8):
    image1, _ = data[index1]
    image2, _ = data[index2]
    image1 = image1.reshape(-1, 28 * 28)
    image2 = image2.reshape(-1, 28 * 28)
    bottleneck1 = model.encode(image1)
    bottleneck2 = model.encode(image2)

    interpolated_tensors = []
    for alpha in torch.linspace(0, 1, n_steps):
        interpolated_bottleneck = alpha * bottleneck1 + (1 - alpha) * bottleneck2
        interpolated_tensors.append(interpolated_bottleneck)

    reconstructed_images = [model.decode(tensor) for tensor in interpolated_tensors]

    plt.figure(figsize=(10, 5))
    for i, image in enumerate(reconstructed_images):
        plt.subplot(1, n_steps, i + 1)
        plt.imshow(image.view(28, 28).detach().numpy(), cmap='gray')
        plt.axis('off')
    plt.show()


def main():

    parser = argparse.ArgumentParser(description="Load a pre-trained model")
    parser.add_argument("-l", "--loadpath", type=str, default="MLP.8.pth",  help="Path to load a pre-trained model")
    args = parser.parse_args()

    data = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    model = torch.load('/Users/markbenhamu/PycharmProjects/CMPE474Lab1/' + args.loadpath)
    model.eval()

    while True:
        # Ask the user for an index number
        index_input = input(
            "Enter an index number to select which images to compare before and after compression(or '000' to go to "
            "the next section): ")

        # Check if the user wants to exit
        if index_input == '000':
            break

        try:
            index = int(index_input)
            # Check if the index is valid
            if index < 0 or index >= len(data):
                print("Invalid index. Please enter a valid index.")
            else:
                print('Let\'s compare: ')
                image_comp(index, model, data)
        except ValueError:
            print("Invalid input. Please enter a valid index or '000' to exit.")

    while True:
        index_input = input(
            "Now lets try it with some noise. Enter an index number to select which image to compare (or '000' to "
            "exit): ")

        # Check if the user wants to exit
        if index_input == '000':
            break

        try:
            index = int(index_input)
            # Check if the index is valid
            if index < 0 or index >= len(data):
                print("Invalid index. Please enter a valid index.")
            else:
                print('Let\'s compare: ')
                reconstruct_and_plot_image(index, model, data)
        except ValueError:
            print("Invalid input. Please enter a valid index or '000' to exit.")

    while True:
        index_input1 = input("Enter the first index for interpolation (or '000' to exit): ")

        if index_input1 == '000':
            break
        index1 = int(index_input1)
        if index1 < 0 or index1 >= len(data):
            print("Invalid index. Please enter a valid index.")
        else:
            index_input2 = input("Enter the second index for interpolation: ")
            index2 = int(index_input2)
            # Check if the index is valid
            if index2 < 0 or index2 >= len(data):
                print("Invalid index. Please enter a valid index.")
            else:
                print('Let\'s interpolate: ')
                interpolate_and_plot(model, index1, index2, data, n_steps=10)


if __name__ == "__main__":
    main()
