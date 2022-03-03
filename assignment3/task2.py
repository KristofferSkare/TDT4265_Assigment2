import pathlib
import matplotlib.pyplot as plt
import torch
import utils
from torch import dropout, nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy
import numpy as np
import json

key = "tryhard"

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
                 activation_function=nn.ReLU(),
                 drop_out=0.0,
                 avg_pool = False,
                 ):          
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
            conv_layer =  nn.Conv2d(
                    in_channels=layer_depth,
                    out_channels=num_filters,
                    kernel_size=kernels[i]["size"],
                    stride=kernels[i]["stride"],
                    padding=kernels[i]["padding"]
                )
           
            conv_layers.append(conv_layer)

            image_shape = new_shape_after_convolution_or_pooling(image_shape, kernels[i]["size"], kernels[i]["stride"], kernels[i]["padding"])
            
            conv_layers.append(activation_function)
            if avg_pool:
                conv_layers.append(nn.AvgPool2d(kernel_size=pooling[i]["size"], stride=pooling[i]["stride"]))
            else:
                conv_layers.append(nn.MaxPool2d(kernel_size=pooling[i]["size"], stride=pooling[i]["stride"]))


            image_shape = new_shape_after_convolution_or_pooling(image_shape, pooling[i]["size"], pooling[i]["stride"])
            
            if (batch_normalization):
                conv_layers.append(nn.BatchNorm2d(num_filters))

            layer_depth = num_filters

        self.feature_extractor = nn.Sequential(*conv_layers)

        num_flattened_nodes = int(image_shape[0] * image_shape[1] * layer_depth)

        hidden_layers = []
        num_input_nodes = num_flattened_nodes
        if drop_out > 0:
            hidden_layers.append(nn.Dropout(p=drop_out))
        
        for num_nodes in hidden_linear_layers:
            layer = nn.Linear(num_input_nodes, num_nodes)
            hidden_layers.append(layer)
            hidden_layers.append(activation_function)
          
            if drop_out > 0:
                hidden_layers.append(nn.Dropout(p=drop_out))

            num_input_nodes = num_nodes
            

        self.num_output_features = num_flattened_nodes
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        last_layer =  nn.Linear(hidden_linear_layers[-1], num_classes)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            *hidden_layers, 
            last_layer
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
    weight_decay = 2e-5
    #learning_rate = 5e-2
    learning_rate = 7e-4
    early_stop_count = 6
    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(
        image_channels=3, 
        num_classes=10, 
        batch_normalization=True, 
        drop_out=0.4,
        #activation_function=nn.ELU(),
        #hidden_linear_layers=[64],
        convolutional_layers=[64,128,256],
        kernels=[{"size": 3, "stride": 1, "padding": 1}, {"size": 3, "stride": 1, "padding": 1}, {"size": 3, "stride": 1, "padding": 1}],
        pooling=[{"size": 4, "stride": 2}, {"size":3, "stride": 2}, {"size": 2, "stride": 2}]
        #pooling=[{"size": 2, "stride": 2}, {"size":2, "stride": 2}, {"size": 2, "stride": 1}],
        #avg_pool=True,
        )
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders,
        weight_decay=weight_decay,
    )
    trainer.train()
    trainer.model.eval()

    train_loss, train_accuracy = compute_loss_and_accuracy(trainer.dataloader_train, trainer.model, trainer.loss_criterion)
    val_loss, val_accuracy = compute_loss_and_accuracy(trainer.dataloader_val, trainer.model, trainer.loss_criterion)
    test_loss, test_accuracy = compute_loss_and_accuracy(trainer.dataloader_test, trainer.model, trainer.loss_criterion)
    
    print(f"Train: \n\tLoss: {train_loss}\n\tAccuracy: {train_accuracy}")
    print(f"Validation: \n\tLoss: {val_loss}\n\tAccuracy: {val_accuracy}")
    print(f"Test: \n\tLoss: {test_loss}\n\tAccuracy: {test_accuracy}")


    data = {}
    with open("models.json", "r") as file:
        data = json.load(file)
    
    data[key] = {
        "train": {"loss": train_loss, "acc": train_accuracy, "history": trainer.train_history},
        "validation": {"loss": val_loss, "acc": val_accuracy, "history": trainer.validation_history},
        "test": {"loss": test_loss, "acc": test_accuracy}
    }

    with open("models.json", "w") as file:
        json.dump(data, file)

    create_plots(trainer, "tryhard")

if __name__ == "__main__":
    main()