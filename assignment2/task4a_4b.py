import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer
import numpy as np


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_weight_init = True
    use_improved_sigmoid = True
    use_momentum = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()

    mean = np.mean(X_train)
    std = np.std(X_train)

    X_train = pre_process_images(X_train, mean, std)
    X_val = pre_process_images(X_val, mean, std)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

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

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!

    use_improved_weight_init = True
    use_improved_sigmoid = True
    use_momentum = False
    neurons_per_layer = [32, 10]


    model_no_shuffle = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_no_shuffle, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_no_shuffle, val_history_no_shuffle = trainer_shuffle.train(
        num_epochs)

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!

    use_improved_weight_init = True
    use_improved_sigmoid = True
    use_momentum = False
    neurons_per_layer = [128, 10]


    model_no_shuffle_1 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_shuffle_1 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_no_shuffle_1, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_no_shuffle_1, val_history_no_shuffle_1 = trainer_shuffle_1.train(
        num_epochs)
        

    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Task 3 Model", npoints_to_average=10)
    utils.plot_loss(
        train_history_no_shuffle["loss"], "Task 4a Model", npoints_to_average=10)
    utils.plot_loss(
        train_history_no_shuffle_1["loss"], "Task 4b Model", npoints_to_average=10)
    plt.ylim([0, 0.6])
    plt.subplot(1, 2, 2)
    plt.ylim([0.92, .97])
    utils.plot_loss(val_history["accuracy"], "Task 3 Model")
    utils.plot_loss(
        val_history_no_shuffle["accuracy"], "Task 4a Model")
    utils.plot_loss(
        val_history_no_shuffle_1["accuracy"], "Task 4b Model")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("task4a_4b.png")
    plt.show()
 