import pathlib
import matplotlib.pyplot as plt
import torch
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer
import numpy as np

def new_shape_after_convolution_or_pooling(image_shape, kernel_size, stride, padding=0):
    dilation = 1
    width, height = image_shape

    width = np.floor((width - dilation*(kernel_size - 1) + 2*padding - 1)/stride + 1)
    height = np.floor((height - dilation*(kernel_size - 1) + 2*padding - 1)/stride + 1)

    return (width, height)

class ExampleModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes,
                 convolutional_layers=[32,64,128],
                 hidden_linear_layers=[64],
                 kernels = [{"size": 5, "stride": 1, "padding": 2}]*3,
                 pooling = [{"size": 2, "stride": 2}]*3,
                 input_image_shape=(32,32),
                 batch_normalization=False,
                 activation_function=nn.ReLU()):          
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()

        self.num_classes = num_classes
        # Define the convolutional layers
        layer_depth = image_channels

        image_shape = input_image_shape

        conv_layers = []
        for i, num_filters in enumerate(convolutional_layers):
            conv_layers.append(
                nn.Conv2d(
                    in_channels=layer_depth,
                    out_channels=num_filters,
                    kernel_size=kernels[i]["size"],
                    stride=kernels[i]["stride"],
                    padding=kernels[i]["padding"]
                )
            )
            image_shape = new_shape_after_convolution_or_pooling(image_shape, kernels[i]["size"], kernels[i]["stride"], kernels[i]["padding"])
            
            conv_layers.append(activation_function)
            
            conv_layers.append(nn.MaxPool2d(kernel_size=pooling[i]["size"], stride=pooling[i]["stride"]))


            image_shape = new_shape_after_convolution_or_pooling(image_shape, pooling[i]["size"], pooling[i]["stride"])
            
            if (batch_normalization):
                conv_layers.append(nn.BatchNorm2d(num_filters))

            layer_depth = num_filters

        self.feature_extractor = nn.Sequential(*conv_layers)

        num_flattened_nodes = int(image_shape[0] * image_shape[1] * layer_depth)

        hidden_layers = []
        num_input_nodes = num_flattened_nodes
        for num_nodes in hidden_linear_layers:
            hidden_layers.append(nn.Linear(num_input_nodes, num_nodes))
            hidden_layers.append(activation_function)
            num_input_nodes = num_nodes

        self.num_output_features = num_flattened_nodes
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            *hidden_layers, 
            nn.Linear(hidden_linear_layers[-1], num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]
        expected_shape = (batch_size, self.num_classes)

        out = self.classifier(self.feature_extractor(x))

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
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(
        image_channels=3, 
        num_classes=10, 
        #batch_normalization=True, 
        #convolutional_layers=[32,64,128],
        #kernels=[{"size": 5, "stride": 1, "padding": 0}, {"size": 3, "stride": 2, "padding": 2}, {"size": 5, "stride": 1, "padding": 2}],
        #pooling=[{"size": 3, "stride": 2}, {"size": 2, "stride": 2}, {"size": 2, "stride": 2}]
        )
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    create_plots(trainer, "task3_1")

if __name__ == "__main__":
    main()