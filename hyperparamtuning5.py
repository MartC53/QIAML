import optuna
from get_data import Dataset_from_directory
import sys
import tensorflow as tf

N_TRAIN_EXAMPLES = 12
N_VALID_EXAMPLES = 6
BATCHSIZE = 3
CLASSES = 3
EPOCHS = 10  # need a model, ideally would run ~30 from prior knowledge,
# computationally too expensive


def create_model(trial):
    # We optimize the numbers of layers,
    # their units and weight decay parameter.
    activation_options = ["elu", "relu", "linear", "selu"]
    activation = trial.suggest_categorical("activation", activation_options)
    model = tf.keras.Sequential(
        [
          tf.keras.layers.Rescaling(1./255, input_shape=(900, 900, 3)),
          tf.keras.layers.Conv2D(16, 3, activation=activation),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(32, activation=activation),
          tf.keras.layers.Dense(CLASSES)
          ]
        )
    return model


def create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.
    kwargs = {}
    optimizer_options = ["RMSprop", "Adam", "SGD"]
    optimizer_selected = trial.suggest_categorical(
        "optimizer",
        optimizer_options
        )
    if optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float(
            "rmsprop_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["decay"] = trial.suggest_float("rmsprop_decay", 0.85, 0.99)
        kwargs["momentum"] = trial.suggest_float(
            "rmsprop_momentum",
            1e-5, 1e-1,
            log=True
            )
    elif optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float(
            "adam_learning_rate",
            1e-5, 1e-1,
            log=True
            )
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["momentum"] = trial.suggest_float(
            "sgd_opt_momentum",
            1e-5, 1e-1,
            log=True
            )

    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    return optimizer


def learn(model, optimizer, dataset, mode="eval"):
    accuracy = tf.metrics.Accuracy("accuracy", dtype=tf.float32)

    for batch, (images, labels) in enumerate(dataset):
        with tf.GradientTape() as tape:
            logits = model(images, training=(mode == "train"))
            loss_value = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits,
                    labels=labels)
            )
            if mode == "eval":
                accuracy(
                    tf.argmax(logits, axis=1, output_type=tf.int64),
                    tf.cast(labels, tf.int64)
                )
            else:
                grads = tape.gradient(loss_value, model.variables)
                optimizer.apply_gradients(zip(grads, model.variables))

    if mode == "eval":
        return accuracy

# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).


def objective(trial):
    # Get picture data.
    data_dir = './Datasets/3_split'
    predict_dir = './Datasets/cropped_jpgs/3_split_test2'
    train_ds, valid_ds, pred_ds = Dataset_from_directory(
        data_dir,
        predict_dir, 0.5
        )

    # Build model and optimizer.
    model = create_model(trial)
    optimizer = create_optimizer(trial)

    # Training and validating cycle.
    with tf.device("/cpu:0"):
        for _ in range(EPOCHS):
            learn(model, optimizer, train_ds, "train")

        accuracy = learn(model, optimizer, valid_ds, "eval")

    # Return last validation accuracy.
    return accuracy.result()


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    # past 3 trials found best in <5 trials
    sys.stdout = open('Hyperparam_output5.txt', 'w')
    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    sys.stdout.close()
