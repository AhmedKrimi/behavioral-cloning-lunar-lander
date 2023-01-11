from __future__ import print_function

import numpy as np
import os
import torch

from agent.bc_agent import BCAgent
from utils import sample_minibatch
from utils import read_data
from utils import preprocessing
from datetime import datetime
from tensorboard_evaluation import Evaluation
from config import Config

SAVE_EVERY = 500
VALIDATION_EVERY_STEPS = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray, config: Config, model_dir: str = "./models", tensorboard_dir: str = "./tensorboard") -> None:
    """
    This method trains the neural network
    """
    # Load training configuration
    n_minibatches, batch_size, lr, hidden_units, history_length, agent_type = config.n_minibatches, config.batch_size, config.lr, config.hidden_units, config.history_length, config.agent_type
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    print("... train model")
    # Specify the agent type with the neural network in agents/bc_agent.py
    agent = BCAgent(agent_type, lr, hidden_units, history_length)
    states = ['training_accuracy', 'validation_accuracy', 'loss']
    tensorboard_eval = Evaluation(
        tensorboard_dir, "tensorboard_events", states)
    # Compute training/validation accuracy and loss for the batch
    for i in range(n_minibatches):
        X_batch, y_batch = sample_minibatch(
            X_train, y_train, batch_size=batch_size, train=True)
        loss = agent.update(X_batch, y_batch)
        if i % VALIDATION_EVERY_STEPS == 0:
            training_correct = 0
            training_total = 0
            validation_correct = 0
            validation_total = 0
            training_accuracy = 0
            validation_accuracy = 0
            with torch.no_grad():
                output_training = agent.predict(
                    torch.tensor(X_batch).to(device))
                output_validation = agent.predict(
                    torch.tensor(X_valid).to(device))
            # Compute training accuracy
            for idx, label in enumerate(output_training):
                if torch.argmax(label) == torch.tensor(y_batch[idx], dtype=torch.long, device=device):
                    training_correct += 1
                training_total += 1
            # Compute validation accuracy
            for idx, label in enumerate(output_validation):
                if torch.argmax(label) == torch.tensor(y_valid[idx], dtype=torch.long, device=device):
                    validation_correct += 1
                validation_total += 1
            training_accuracy = training_correct/training_total
            validation_accuracy = validation_correct/validation_total
            print("Episode %d of %d" % (i, n_minibatches))
            print("Training accuracy: %f" % training_accuracy)
            print("Validation accuracy: %f" % validation_accuracy)
            # Compute training/ validation accuracy and write it to tensorboard
            eval_dic = {'training_accuracy': training_accuracy,
                        'validation_accuracy': validation_accuracy, 'loss': loss.item()}
            tensorboard_eval.write_episode_data(i, eval_dic)
        # Save the agent every SAVE_EVERY
        if i % SAVE_EVERY == 0:
            save_file = datetime.now().strftime("%Y%m%d-%H%M%S") + "_bc_agent.pt"
            model_dir = agent.save(os.path.join("./models", save_file))
            print("Model saved in file: %s" % model_dir)

if __name__ == "__main__":
    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data")
    # Load training configuration
    conf = Config()
    # preprocess data
    if conf.agent_type == "CNN":
        X_train, y_train, X_valid, y_valid = preprocessing(
            X_train, y_train, X_valid, y_valid, conf)
    # train model
    train_model(X_train, y_train, X_valid, y_valid, conf)
