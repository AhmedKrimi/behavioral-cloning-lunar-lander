import pickle
import gzip
import glob
import os
import numpy as np
import random as rnd
from config import Config
from typing import Union


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """ 
    this method converts rgb images to grayscale arrays and normalizes the images
    """
    gray = np.dot(rgb[..., :3], [0.2125, 0.7154, 0.0721])
    gray = 2 * gray.astype('float32') - 1
    return gray

def read_data(datasets_dir: str = "./data", frac: float = 0.01) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This method reads the data saved while playing and transform it to training set and validation set
    """
    print("... read data")
    file_names = glob.glob(os.path.join(datasets_dir, "*.gzip"))
    print(file_names)
    X = []
    y = []
    n_samples = 0
    for data_file in file_names:
        f = gzip.open(data_file, 'rb')
        data = pickle.load(f)
        n_samples += len(data["state"])
        # Access images use state_img
        X.extend(data["state_img"])
        y.extend(data["action"])
    X = np.array(X).astype('float32')
    y = np.array(y).astype('float32')
    # split data into training and validation set
    X_train, y_train = X[:int((1-frac) * n_samples)
                         ], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid

def preprocessing(X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray, conf: Config) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This method convert images to grayscale for training and implement the history with skip frames
    """
    # Convert the images in X_train/X_valid to gray scale
    X_train = np.array([rgb2gray(img).reshape(1, 100, 150) for img in X_train])
    X_valid = np.array([rgb2gray(img).reshape(1, 100, 150) for img in X_valid])
    # Adding a history of the last N images to your state so that a state has shape (100, 150, N) with skipping frames in between
    skip_frames, history_length = conf.skip_frames, conf.history_length
    X_tr, y_tr = add_history_skip_frames(
        X_train, y_train, history_length, skip_frames)
    X_val, y_val = add_history_skip_frames(
        X_valid, y_valid, history_length, skip_frames)
    return X_tr, y_tr, X_val, y_val

def sample_minibatch(X: np.ndarray, y: np.ndarray, batch_size: int, train: bool = True) -> Union[np.ndarray, np.ndarray]:
    """
    This method samples minibatches from the training set
    """
    length = X.shape[0]
    X_batch = []
    y_batch = []
    if train:
        sampled_idx = rnd.sample(range(1, length), batch_size)
    else:
        sampled_idx = rnd.sample(range(1, length), 2*batch_size)
    for idx in sampled_idx:
        X_batch.append(X[idx])
        y_batch.append(y[idx])
    return X_batch, y_batch

def add_history_skip_frames(X: np.ndarray, y: np.ndarray, hist_len: int, n_skip: int) -> Union[np.ndarray, np.ndarray]:
    """
    This methods implement the history (N images as one input) and skip frames in between
    """
    length = len(X)
    idx = 0
    X_seq = []
    y_seq = []
    while idx + n_skip < length - 1:
        X_tmp = []
        s = (1, 100, 150)
        y_tmp = np.zeros(s)
        while len(X_tmp) < hist_len:
            X_tmp.append(X[idx][0])
            y_tmp = y[idx]
            if idx + n_skip < length - 1:
                idx = idx + n_skip + 1
            else:
                X_tmp = []
                break
        if len(X_tmp) != 0:
            X_tmp = np.array(X_tmp).astype(
                'float32').reshape(hist_len, 100, 150)
            X_seq.append(X_tmp)
            y_seq.append(y_tmp)

    X_seq = np.array(X_seq).astype('float32')
    y_seq = np.array(y_seq).astype('float32')
    return X_seq, y_seq
