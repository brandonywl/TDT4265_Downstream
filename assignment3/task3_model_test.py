import pathlib
import matplotlib.pyplot as plt
import utils
import torch
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer


class Task3_Model_test(nn.Module):
    
    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes

        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(128),
        )

        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32768, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        out = self.conv_stack(x)
        out = self.fc_stack(out)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


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
    epochs = 100
    batch_size = 64
    learning_rate = 1e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    optimizer = torch.optim.SGD
    task3_model_test = Task3_Model_test(image_channels=3, num_classes=10)
    task3_trainer_test = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        task3_model_test,
        dataloaders,
        optimizer
    )
    task3_trainer_test.train()
    create_plots(task3_trainer_test, "task3a_model_test")

    from task3_model_1 import Task3_Model_1

    task3_model_1 = Task3_Model_1(image_channels=3, num_classes=10)
    task3_trainer_1 = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        task3_model_1,
        dataloaders,
        optimizer
    )
    task3_trainer_1.train()

    utils.plot_loss(task3_trainer_test.train_history["loss"], "3 Conv Layer - Train")
    utils.plot_loss(task3_trainer_test.validation_history["loss"], "3 Conv Layer - Validation")

    utils.plot_loss(task3_trainer_1.train_history["loss"], "6 Conv Layer - Train")
    utils.plot_loss(task3_trainer_1.validation_history["loss"], "6 Conv Layer - Validation")

    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Training Step")
    plt.title("Comparison plot of Loss given different number of convolution layers")

    plt.savefig("task3d.png")
    plt.show()



if __name__ == "__main__":
    main()