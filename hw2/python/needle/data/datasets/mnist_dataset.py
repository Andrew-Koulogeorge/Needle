from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct
import gzip


BIG_ENDIAN = ">" # string format for reading bytes in big endian style
MAX_INTENSITY = 255 # largest intensity of pixel in image

class MNISTDataset(Dataset):
    def __init__(self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,):

        images = []
        with gzip.open(image_filename, "rb") as file:
          header = file.read(16)
          magic_number, num_images, n_rows, n_cols = struct.unpack(">IIII", header)
          print(magic_number)
          if(magic_number != 2051):
            print("Invalid Magic Number\n")
            return 
          image_size = n_rows*n_cols

          for _ in range(num_images):
            # read in each pixel and append it to an image
            image = list(file.read(image_size))
            images.append(image)
              
        with gzip.open(label_filename, "rb") as file:
          header = file.read(8)
          magic_number, num_images = struct.unpack(BIG_ENDIAN + "II", header)
          labels = list(file.read(num_images))
          
        images_np = np.array(images, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.uint8)

        # ensure types in X and Y and return
        images_np /= MAX_INTENSITY

        self.data = images_np
        print(self.data.shape)
        self.labels = labels_np
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION

        data, labels = self.data[index], self.labels[index]
        if self.transforms:
            data = data.reshape((28, 28, -1)) # prepare data for transformation
            data = self.apply_transforms(data) # apply transform function that expects H x W x C
            return data.reshape(-1, 28 * 28), labels # reform data back into flattened form for feed forward network
        return data, labels
            

        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.data.shape[0]
        ### END YOUR SOLUTION