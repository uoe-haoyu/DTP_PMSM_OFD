import torch
import numpy as np
from torch.utils.data import Dataset


# Define external functions for noise and scaling
def random_add_gaussian(data, lambda_value=0.1):
    # Generate Gaussian noise with mean 0 and variance lambda
    epsilon = torch.normal(mean=0, std=lambda_value, size=data.shape).numpy()
    return data + epsilon

def random_scale(data, lambda_value=0.1):
    # Generate a scaling factor from Gaussian with mean 1 and variance lambda
    sigma = torch.normal(mean=1, std=lambda_value, size=(data.shape[0], 1)).numpy()
    return sigma * data

def apply_random_noise_and_scaling(data, lambda_value=0.1,pro=0.2):
    # Apply RandomAddGaussian with 20% probability
    if torch.rand(1).item() > (1-pro):
        data = random_add_gaussian(data, lambda_value)

    # # Apply RandomScale with 20% probability
    if torch.rand(1).item() > (1-pro):
        data = random_scale(data, lambda_value)

    return data

class MyDataset(Dataset):
    """
    Dataset
    """

    def __init__(self, csv_path, transform=None, loader=None, is_val=False, train=True):

        super(MyDataset, self).__init__()
        if csv_path is not None:
            df_train = np.load(csv_path, allow_pickle=True)
        else:
            df_train = None

        self.train = train
        self.df = df_train
        self.loader = loader
        if csv_path is not None:
            print('The current dataset length is:', self.__len__())


    def __getitem__(self, index):

        input = self.df[0]
        label = self.df[1]
        input = input[:,index,:]
        label = label[index]

        if self.train:
            input = apply_random_noise_and_scaling(input,lambda_value=0.1)
        input = torch.Tensor(input)
        label = torch.tensor([label], dtype=torch.long)
        label = label.squeeze(0)
        return input, label

    def __len__(self):
        if self.df is not None:
            return (self.df[0].shape[1])

def get_data(train_path, test_path, rate=0.1, is_val=False):

    traindata = MyDataset(train_path,train=False)
    testdata = MyDataset(test_path,train=False)

    if is_val:
        valiation = MyDataset(is_val,train=False)
        return {'train': traindata, 'test': testdata, 'val':valiation}

    else:
        print('没有分割验证集合，只有训练集和测试机')
        return {'train': traindata, 'test': testdata}

#To perform robustness testing with training enabled (train=True).
def get_pathdata(test_path):
    return MyDataset(test_path,train=False)


