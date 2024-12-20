# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: Abstract Base Class for all types of deepfake datasets.

import sys
sys.path.append('.')

import os
import math
import yaml
import glob
import json
import csv
import pandas as pd

import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image
from collections import defaultdict

from concurrent.futures import ThreadPoolExecutor

import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T

import albumentations as A

from training.dataset.albu import IsotropicResize


class DeepfakeAbstractBaseDataset(data.Dataset):
    """
    Abstract base class for all deepfake datasets.
    """
    def __init__(self, config=None, mode='train'):
        """Initializes the dataset object.

        Args:
            config (dict): A dictionary containing configuration parameters.
            mode (str): A string indicating the mode (train or test).

        Raises:
            NotImplementedError: If mode is not train or test.
        """
        
        # Set the configuration and mode
        self.config = config
        self.mode = mode
        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]

        # Dataset dictionary
        self.image_list = []
        self.label_list = []
        
        # Set the dataset dictionary based on the mode
        if mode == 'train':
            dataset_list = config['train_dataset']
            # Training data should be collected together for training
            image_list, label_list = [], []
            for one_data in dataset_list:
                tmp_image, tmp_label = self.collect_img_and_label_for_one_dataset(one_data)
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
        elif mode == 'test':
            dataset_list = config['test_dataset']
            for one_data in dataset_list:
            # Test dataset should be evaluated separately. So collect only one dataset each time
                image_list, label_list = self.collect_img_and_label_for_one_dataset(one_data)
        else:
            raise NotImplementedError('Only train and test modes are supported.')

        assert len(image_list)!=0 and len(label_list)!=0, f"Collect nothing for {mode} mode!"
        self.image_list, self.label_list = image_list, label_list
                    
        # Create a dictionary containing the image and label lists
        self.data_dict = {
            'image': self.image_list, 
            'label': self.label_list, 
        }
        
        self.transform = self.init_data_aug_method()
        
    def init_data_aug_method(self):
        trans = A.Compose([           
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            A.OneOf([                
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'], contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'], quality_upper=self.config['data_aug']['quality_upper'], p=0.5)
        ], 
            keypoint_params=A.KeypointParams(format='xy') if self.config['with_landmark'] else None
        )
        return trans
       
    def data_aug(self, img, landmark=None, mask=None, augmentation_seed=None):
        """
        Apply data augmentation to an image, landmark, and mask.
    
        Args:
            img: An Image object containing the image to be augmented.
            landmark: A numpy array containing the 2D facial landmarks to be augmented.
            mask: A numpy array containing the binary mask to be augmented.
    
        Returns:
            The augmented image, landmark, and mask.
        """
    
        # Set the seed for the random number generator
        if augmentation_seed is not None:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)
    
        # Create a dictionary of arguments
        kwargs = {'image': img}
        
        # Apply data augmentation
        transformed = self.transform(**kwargs)
        
        # Get the augmented image
        augmented_img = transformed['image']
        
        # Add Gaussian noise to the augmented image
        #augmented_img = self.add_gaussian_noise(augmented_img, 0, 15)  # Adjust std as necessary
    
        # Reset the seeds to ensure different transformations for different videos
        if augmentation_seed is not None:
            random.seed()
            np.random.seed()
    
        return augmented_img
     
    def add_gaussian_noise(self,image, mean, std):
        """Adds Gaussian noise to an image.
        
        Args:
            image: A PIL Image object.
            mean: Mean of the Gaussian noise.
            std: Standard deviation of the Gaussian noise.
    
        Returns:
            A PIL Image object with added noise.
        """
        # Convert image to numpy array
        img_array = image
    
        # Generate Gaussian noise
        noise = np.random.normal(mean, std, img_array.shape).astype('uint8')
    
        # Add noise to the image and clip the values to stay in valid range
        noisy_img_array = np.clip(img_array + noise, 0, 255)
    
        # Convert back to PIL Image
        return Image.fromarray(noisy_img_array)

 
    '''def verify_image(self, image_path):
        """Verifies if the image is valid."""
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify that it's a valid image
            return True
        except (IOError, FileNotFoundError):
            return False
        
    def collect_img_and_label_for_one_dataset(self, dataset_name: str):
        """Collects image and label lists.
    
        Args:
            dataset_name (str): A list containing one dataset information. e.g., 'FF-F2F'
    
        Returns:
            list: A list of image paths.
            list: A list of labels.
    
        Raises:
            ValueError: If image paths or labels are not found.
            NotImplementedError: If the dataset is not implemented yet.
        """
        label_list = []
        frame_path_list = []
        root_dir = ""  # Set your root directory if needed
        
        # Class mapping
        class_mapping = {
            'real': 0,
            'DeepFakes': 1,
            'Face2Face': 2,
            'FaceSwap': 3,
            'NeuralTextures': 4
        }
        
        # Combined file path
        if self.mode == 'test':
            file_path = 'datasets/selected_image_paths_1_test.csv'
        else:
            file_path = 'datasets/selected_image_paths_1_t_v.csv'
    
        # Read all lines at once
        with open(file_path, 'r') as file:
            lines = file.readlines()
    
        # Filter lines relevant to the dataset
        relevant_lines = [line for line in lines if dataset_name in line]
    
        # Check file existence and collect paths and labels
        for line in relevant_lines:
            image_path = os.path.join(root_dir, line.strip())
            
            # Verify if the image is valid
            if self.verify_image(image_path):
                # Determine the label based on the presence of keywords
                label = None
                for keyword, class_label in class_mapping.items():
                    if keyword in line:
                        label = class_label
                        break
                
                if label is not None:
                    frame_path_list.append(image_path)
                    label_list.append(label)
                else:
                    print(f"Label not found for: {image_path}")  # Or log this error
            else:
                print(f"Invalid image file: {image_path}")  # Or log this error
    
        # Shuffle the lists together
        if frame_path_list and label_list:
            shuffled = list(zip(label_list, frame_path_list))
            random.shuffle(shuffled)
            label_list, frame_path_list = zip(*shuffled)
    
        return frame_path_list, label_list
     '''

    def verify_image(self, image_path):
         """Verifies if the image is valid."""
         try:
             with Image.open(image_path) as img:
                 img.verify()  # Verify that it's a valid image
             return True
         except (IOError, FileNotFoundError):
             return False

    def collect_img_and_label_for_one_dataset(self, dataset_name: str):
        """Collects image and label lists."""

        label_list = []
        frame_path_list = []
        root_dir = ""  # Set your root directory if needed
        
        # Class mapping
        class_mapping = {
            'real': 0,
            'DeepFakes': 1,
            'Face2Face': 2,
            'FaceSwap': 3,
            'NeuralTextures': 4
        }

        # Combined file path
        file_path = 'datasets/all_image_paths_new_test.csv' if self.mode == 'test' else 'datasets/selected_image_paths_t_v_15.csv'
        
        # Check file existence and collect paths and labels
        with open(file_path, 'r') as file:
            # Use ThreadPoolExecutor for parallel verification
            with ThreadPoolExecutor() as executor:
                futures = {}
                for line in file:
                    if dataset_name in line:
                        image_path = os.path.join(root_dir, line.strip())
                        futures[executor.submit(self.verify_image, image_path)] = image_path

                for future in futures:
                    image_path = futures[future]
                    if future.result():  # Check if the image is valid
                        # Determine the label based on the presence of keywords
                        label = next((class_label for keyword, class_label in class_mapping.items() if keyword in image_path), None)
                        if label is not None:
                            frame_path_list.append(image_path)
                            label_list.append(label)
                        else:
                            print(f"Label not found for: {image_path}")
                    else:
                        print(f"Invalid image file: {image_path}")
        
        # Shuffle the lists together
        if frame_path_list and label_list:
            shuffled = list(zip(label_list, frame_path_list))
            random.shuffle(shuffled)
            label_list, frame_path_list = zip(*shuffled)

        return frame_path_list, label_list



     
    def load_rgb(self, file_path):
        """
        Load an RGB image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the image file.

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        size = 256
        img = cv2.imread(file_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))


    def to_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.
        """
        return T.ToTensor()(img)

    def normalize(self, img):
        """
        Normalize an image.
        """
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def __getitem__(self, index):
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Get the image paths and label
        image_path = self.data_dict['image'][index]
        label = self.data_dict['label'][index]
        # Load the image
        image = self.load_rgb(image_path)
        image = np.array(image)  # Convert to numpy array for data augmentation
        # To tensor and normalize
        image_trans = self.normalize(self.to_tensor(image))
        return image_trans, label
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                          the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors
        images, labels = zip(*batch)
        # Stack the image, label, landmark, and mask tensors
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        return data_dict

    def __len__(self):
        """
        Return the length of the dataset.

        Args:
            None.

        Returns:
            An integer indicating the length of the dataset.

        Raises:
            AssertionError: If the number of images and labels in the dataset are not equal.
        """
        assert len(self.image_list) == len(self.label_list), 'Number of images and labels are not equal'
        return len(self.image_list)

if __name__ == "__main__":
    with open('training/config/detector/ucf.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_set = DeepfakeAbstractBaseDataset(
                config = config,
                mode = 'test', 
            )
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True, 
            num_workers=int(config['workers']),
            collate_fn=train_set.collate_fn,
        )


    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        print(batch)

    