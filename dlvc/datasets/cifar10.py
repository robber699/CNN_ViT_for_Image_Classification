import numpy as np
import pickle
import tarfile
from dlvc.datasets.dataset import ClassificationDataset
from typing import Tuple
import matplotlib.pyplot as plt



class CIFAR10Dataset(ClassificationDataset):
    '''
    Custom CIFAR-10 Dataset.
    '''

    def __init__(self, fdir: str, subset: str, transform=None):
        '''
        Loads the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all images from "data_batch_5".
          - The test set contains all images from "test_batch".

        Images are loaded in the order they appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in RGB channel order.
        '''
        if subset == 'training':
            target_files =[f'data_batch_{i}' for i in range(1, 4)] #['data_batch_1']
        elif subset == 'validation':
            target_files = ['data_batch_5']
        elif subset == 'test':
            target_files = ['test_batch']
        else:
            raise ValueError("Invalid subset")

        self.data = []
        self.labels = []

        # Load data from tar file
        with tarfile.open(fdir, 'r') as tar:
            for target_file in target_files:
                batch_data = self.unpickle_from_tar(tar, f'cifar-10-batches-py/{target_file}')
                self.data.extend(batch_data[b'data'])
                self.labels.extend(batch_data[b'labels'])

        # Convert to numpy arrays
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.transform = transform

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple:
        '''
        Returns the idx-th sample in the dataset, which is a tuple,
        consisting of the image and labels.
        Applies transforms if not None.
        Raises IndexError if the index is out of bounds.
        '''
        if idx >= len(self):
            raise IndexError("Index out of bounds")

        img = self.data[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''
        return len(self.classes)

    @staticmethod
    def unpickle_from_tar(tar, target_file):
        '''
        Helper function to unpickle data from a tarfile.
        '''
        file = tar.extractfile(target_file)
        data_dict = pickle.load(file, encoding='bytes')
        return data_dict


'''
# Instantiate CIFAR10Dataset for training, validation, and test sets
train_dataset = CIFAR10Dataset('C:/Users/Admin/Desktop/dlvc/datasets/cifar-10-python.tar.gz', "training")
val_dataset = CIFAR10Dataset('C:/Users/Admin/Desktop/dlvc/datasets/cifar-10-python.tar.gz', "validation")
test_dataset = CIFAR10Dataset('C:/Users/Admin/Desktop/dlvc/datasets/cifar-10-python.tar.gz', "test")

#dataset information
print("Number of samples in training set:", len(train_dataset))
print("Number of samples in validation set:", len(val_dataset))
print("Number of samples in test set:", len(test_dataset))
print("Number of classes:", train_dataset.num_classes())

# Access and visualize some samples
sample_idx = 0
num_samples_to_visualize = 5
for i in range(sample_idx, sample_idx + num_samples_to_visualize):
    image, label = train_dataset[i]
    print("Sample", i, "Label:", label, "Class:", train_dataset.classes[label])
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# __len__ method
print("Length of training dataset:", train_dataset.__len__())

#  num_classes method
print("Number of classes:", train_dataset.num_classes())

#  the data structure of train_dataset
print(train_dataset)

#  the data structure of train_dataset
print("Data structure of train_dataset:")
for idx in (1,3):
    image, label = train_dataset[idx]
    print("Index:", idx)
    print("Image shape:", image.shape)
    print("Label:", label)
'''
