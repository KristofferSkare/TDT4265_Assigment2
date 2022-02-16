import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = True
    use_momentum = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)
    # TODO: Some of the plots needs to be changed before running with 50 epochs, check assignment text
    network_topologies = [[64, 10],[*([64]*10), 10]]
    settings = [[]]
    loss_histories = []
    acc_histories = []
    for neurons_per_layer in network_topologies:
        
        model = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
            
        trainer = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )
        train_history, val_history = trainer.train(num_epochs)
        loss_histories.append({"train": train_history["loss"], "val": val_history["loss"]})
        acc_histories.append({"train": train_history["accuracy"], "val": val_history["accuracy"]})
        print("Training ", str(neurons_per_layer), " done")
    
    plt.suptitle("Loss: 1 hidden layer vs 10 hidden layers")
    runs= [" (1 hidden layer)", " (10 hidden layers)"]
    for i in range(len(network_topologies)):
        utils.plot_loss(loss_histories[i]["train"],
                        "Train" + runs[i], npoints_to_average=10)
    for i in range(len(network_topologies)):
        utils.plot_loss(
            loss_histories[i]["val"], "Val"  + runs[i])
        plt.legend()
    plt.savefig("task4e_loss.png") 
    plt.clf()

    plt.suptitle("Accuracy: 1 hidden layer vs 10 hidden layers")
    for i in range(len(network_topologies)):
        utils.plot_loss(acc_histories[i]["train"], "Train"  + runs[i])
    for i in range(len(network_topologies)):
        utils.plot_loss(
            acc_histories[i]["val"], "Val"  + runs[i])
        plt.legend()

    plt.savefig("task4e_accuracy.png") 
       
