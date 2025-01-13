import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        
        # read article about how to read in data
        # read in the data from the base folder based on train/test flag
        # div pixel values, store in self.X, self.y (numpy array)
        # X should be of shape n x 3 x 32 x 32; y shape n (values between 1-10)
        if train:
            data_batched = []
            labels_batched = []
            for batch in range(1,6):
                file_path = base_folder + "/data_batch_" + str(batch)
                with open(file_path, 'rb') as fo:
                    info = pickle.load(fo, encoding='latin1')                    
                    data, labels = info["data"]/255, info["labels"] # div pixel values
                    data_batched.append(data.reshape(-1,3,32,32))
                    labels_batched.append(labels)
            self.X = np.concatenate(data_batched, axis=0)
            self.y = np.concatenate(labels_batched, axis=0)
        else:
            file_path = base_folder + "/test_batch" 
            with open(file_path, 'rb') as fo:
                info = pickle.load(fo, encoding='latin1')
                data, labels = info["data"]/255, info["labels"] # div pixel values
                self.X = data.reshape(-1,3,32,32)
                self.y = np.array(labels)
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        return self.X[index], self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.y.shape[0]
        ### END YOUR SOLUTION