from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import torchvision.transforms as transforms
import os
import imageio
import numpy as np
import torch
import random
# from PIL import Image

def count_volume(data_sz, vol_sz, stride):
    return 1 + np.ceil((data_sz - vol_sz) / stride.astype(float)).astype(int)

def crop_volume(data, sz, st=(0, 0, 0)):  # C*D*W*H, C=1
    st = np.array(st).astype(np.int32)
    return data[st[0]:st[0]+sz[0], st[1]:st[1]+sz[1], st[2]:st[2]+sz[2]]

class VolumeDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--input_size', nargs='+',type=int, default=[96,96,96], help="Model's input size")
        parser.add_argument('--stride', nargs='+',type=int, help="Stride between subvolumes selection", required=True)
#         self.initialized = True
        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load volumes paths from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load volumes paths from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the # of volumes dataset A
        self.B_size = len(self.B_paths)  # get the # of volumes dataset B
        # Load the volumes
        self.A_data = [imageio.volread(self.A_paths[x]) for x in range(self.A_size)]
        self.B_data = [imageio.volread(self.B_paths[x]) for x in range(self.B_size)]
        # Get volumes sizes
        self.A_input_size = [np.array(x.shape) for x in self.A_data]  
        self.B_input_size = [np.array(x.shape) for x in self.B_data] 
        self.sample_volume_size = np.array(opt.input_size).astype(int)  # model input size
        
        # compute number of samples for each dataset (multi-volume input)
        self.sample_stride = np.array(opt.stride, dtype=int)
        
        self.A_sample_size = [count_volume(self.A_input_size[x], self.sample_volume_size, self.sample_stride) for x in range(len(self.A_input_size))]
        self.B_sample_size = [count_volume(self.B_input_size[x], self.sample_volume_size, self.sample_stride) for x in range(len(self.B_input_size))]

        # total number of possible positions for each volume
        self.A_sample_num = np.array([np.prod(x) for x in self.A_sample_size])
        self.B_sample_num = np.array([np.prod(x) for x in self.B_sample_size])
        self.A_sample_num_a = np.sum(self.A_sample_num)
        self.B_sample_num_a = np.sum(self.B_sample_num)
        self.A_sample_num_c = np.cumsum([0] + list(self.A_sample_num))
        self.B_sample_num_c = np.cumsum([0] + list(self.B_sample_num))
        self.transform = transforms.Compose(transforms.Normalize((0.5,), (0.5,)))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        pos_A, pos_B = self._get_pos_train(self.sample_volume_size)
#         out_A = torch.from_numpy((crop_volume(self.A_data[pos_A[0]], self.sample_volume_size, pos_A[1:])/255.0).astype(np.float32)).unsqueeze(0) # Add batch dimension
#         out_B = torch.from_numpy((crop_volume(self.B_data[pos_B[0]], self.sample_volume_size, pos_B[1:])/255.0).astype(np.float32)).unsqueeze(0)
        out_A = torch.from_numpy((crop_volume(self.A_data[pos_A[0]], self.sample_volume_size, pos_A[1:])/255.0).astype(np.float32))
        # Normalize between -1 and 1 for tanh generator layer
        out_A = ((out_A - 0.5)/0.5).unsqueeze(0)
        out_B = torch.from_numpy((crop_volume(self.B_data[pos_B[0]], self.sample_volume_size, pos_B[1:])/255.0).astype(np.float32))
        out_B = ((out_B - 0.5)/0.5).unsqueeze(0)
        return {'A': out_A, 'B': out_B, 'A_paths': self.A_paths, 'B_paths': self.B_paths}

    def __len__(self):
        """Return the max total number of possible positions in both volumes"""
        return max(self.A_sample_num_a, self.B_sample_num_a)
    
    def _index_to_dataset(self, index_A, index_B):
        # Where index value is inferior than total cummulative positions available, choose that dataset
        return np.argmax(index_A < self.A_sample_num_c) - 1, np.argmax(index_B < self.B_sample_num_c) - 1  # which dataset for A and B
    
    def _get_pos_train(self, vol_size):
        # random: multithread
        # np.random: same seed
        # First value is dataset choice, other values are starting vol postion values x,y,z
        pos_A = [0, 0, 0, 0]
        pos_B = [0, 0, 0, 0]
        # pick a dataset id
        did_A, did_B = self._index_to_dataset(random.randint(0,self.A_sample_num_a-1),random.randint(0,self.B_sample_num_a-1))
        pos_A[0] = did_A
        pos_B[0] = did_B
        # pick a position
        # Leave enough space to get vol_size however through count volume
        tmp_size_A = count_volume(self.A_input_size[did_A], vol_size, self.sample_stride)
        tmp_size_B = count_volume(self.B_input_size[did_B], vol_size, self.sample_stride)
        tmp_pos_A = [random.randint(0,tmp_size_A[x]-1) * self.sample_stride[x] for x in range(len(tmp_size_A))]
        tmp_pos_B = [random.randint(0,tmp_size_B[x]-1) * self.sample_stride[x] for x in range(len(tmp_size_B))]
        pos_A[1:] = tmp_pos_A
        pos_B[1:] = tmp_pos_B
        return pos_A, pos_B

if __name__ == '__main__':
    from options.train_options import TrainOptions
    opt = TrainOptions().parse()
    dataset = VolumeDataset(opt)
    print(dataset.dir_B, dataset.B_paths, dataset.B_size)
    print(dataset.B_input_size, dataset.sample_volume_size)
    print(dataset[1000]['B'].shape)
    print(dataset[2000]['A'].shape)
    from matplotlib import pyplot as plt
    plt.imsave("./t.png",dataset[20]['B'][0,0] ,cmap="gray")