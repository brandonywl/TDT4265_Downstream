import pathlib
import matplotlib.pyplot as plt
import utils
import torch
from torch import nn
from dataloaders import load_cifar10_image_net
from trainer import Trainer
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

class Task4_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10) # No need to apply softmax,
        # as this is done in nn.CrossEntropyLoss
        for param in self.model.parameters(): # Freeze all parameters
          param.requires_grad = False
        for param in self.model.fc.parameters(): # Unfreeze the last fully-connected
          param.requires_grad = True # layer
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional
          param.requires_grad = True # layers
    def forward(self, x):
        x = self.model(x)
        return x


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 5
    task4_batch_size = 32
    task4_learning_rate = 5e-5 # Should be 5e-5 for LeNet
    early_stop_count = 4
    dataloaders = load_cifar10_image_net(task4_batch_size)
    optimizer = torch.optim.Adam
    task4_model = Task4_Model()
    task4_trainer = Trainer(
        task4_batch_size,
        task4_learning_rate,
        early_stop_count,
        epochs,
        task4_model,
        dataloaders,
        optimizer
    )
    task4_trainer.train()
    create_plots(task4_trainer, "task4")


    def torch_image_to_numpy(image: torch.Tensor):
        """
        Function to transform a pytorch tensor to numpy image
        Args:
            image: shape=[3, height, width]
        Returns:
            iamge: shape=[height, width, 3] in the range [0, 1]
        """
        # Normalize to [0 - 1.0]
        image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
        image = image - image.min()
        image = image / image.max()
        image = image.numpy()
        if len(image.shape) == 2: # Grayscale image, can just return
            return image
        assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
        image = np.moveaxis(image, 0, 2)
        return image

    # Task 4b Implementation
    zebra_img = Image.open("images/zebra.jpg") # Open the zebra image in PIL
    zebra_tensor = transforms.ToTensor()(zebra_img).unsqueeze_(0)

    indexes_of_interest = [14, 26, 32, 49, 52]

    feature_map_activations = torchvision.models.resnet18(pretrained=True).conv1(zebra_tensor)
    feature_maps_of_interest = feature_map_activations[0, indexes_of_interest, :, :]

    
    weights_of_interest = torchvision.models.resnet18(pretrained=True).conv1.weight[indexes_of_interest, :, :, :]

    fig, ax = plt.subplots(1, 5, figsize=(20, 5))

    for idx, fm in enumerate(feature_maps_of_interest):
        ax[0, idx].set_title(f"Feature Idx {indexes_of_interest[idx]}")
        x = weights_of_interest[idx]
        x = torch_image_to_numpy(x)
        y = fm.detach().numpy()
        ax[0, idx].imshow(y)
        ax[1, idx].imshow(x)

    plt.savefig('task4b.png')
    plt.show()

    # Task 4c implementation
    zebra_cuda = zebra_tensor.cuda()
    x = zebra_cuda

    for block in task4_model.children():
      for idx, child in enumerate(block.children()):
        if idx == 8:
            break
        x = child(x)
    
    n = 10

    idx_of_interest = [i for i in range(n)]
    fms = x[0, idx_of_interest, :, :]

    fig, ax = plt.subplots(2, n//2, figsize=(20, 8))

    for idx, fm in enumerate(fms):
        ax[idx//5, idx % (n//2)].set_title(f"Feature Idx {idx_of_interest[idx]}")
        y = fm.cpu().detach().numpy()
        ax[idx//5, idx % (n//2)].imshow(y)

    plt.savefig("task4c.png")
    plt.show()



if __name__ == "__main__":
    main()