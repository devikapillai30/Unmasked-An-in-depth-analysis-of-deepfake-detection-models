import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pickle
import os
import pandas as pd
from PIL import Image
import random


class ImageDataset_Train(Dataset):
 

    def __init__(self, csv_file, owntransforms):
        super(ImageDataset_Train, self).__init__()
        self.img_path_label = pd.read_csv(csv_file)
        self.transform = owntransforms

    def __len__(self):
        return len(self.img_path_label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        img_path = self.img_path_label.iloc[idx, 0]

        if not isinstance(img_path, str) or not os.path.exists(img_path):
            raise ValueError(f"Expected img_path to be a valid file path, got {type(img_path)}: {img_path}")



        if img_path != 'Image Path':
            img = Image.open(img_path)
            img = self.transform(img)
           
            label = np.array(self.img_path_label.loc[idx, 'Target'])
            intersec_label = np.array(self.img_path_label.loc[idx, 'Intersection'])

        return {'image': img, 'label': label, 'intersec_label': intersec_label}

class ImageDataset_Test_Ind(Dataset):


    def __init__(self, csv_file, owntransforms):
        super(ImageDataset_Test_Ind, self).__init__()
        self.img_path_label = pd.read_csv(csv_file)
        self.transform = owntransforms

    def __len__(self):
        return len(self.img_path_label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_path_label.iloc[idx, 0]

        if img_path != 'Image Path':
            img = Image.open(img_path)
            img = self.transform(img)
           
            label = np.array(self.img_path_label.loc[idx, 'Target'])

        return {'image': img, 'label': label}


class ImageDataset_Test(Dataset):
    def __init__(self, csv_file, attribute, owntransforms):
        self.transform = owntransforms
        self.img = []
        self.label = []
        
        # Mapping from attribute strings to (intersec_label, age_label) tuples
        # Note: if an attribute doesn't correspond to an age label, we use None
        attribute_to_labels = {
            # Gender, Race, Hair Color
            'male,asian': (0, None, None,None),    # Gender: male, Race: asian, Hair color: None
            'male,white': (1, None, None,None),    # Gender: male, Race: white, Hair color: None
            'male,black': (2, None, None, None),    # Gender: male, Race: black, Hair color: None
            'male,others': (3, None, None, None),   # Gender: male, Race: others, Hair color: None
            'nonmale,asian': (4, None, None, None), # Gender: nonmale, Race: asian, Hair color: None
            'nonmale,white': (5, None, None, None), # Gender: nonmale, Race: white, Hair color: None
            'nonmale,black': (6, None, None, None), # Gender: nonmale, Race: black, Hair color: None
            'nonmale,others': (7, None, None, None),# Gender: nonmale, Race: others, Hair color: None 
            
            # Age, Gender and Race values with Hair Color None
            'young': (None, 0, None, None),         # Age: young, Hair color: None
            'middle': (None, 1, None, None),        # Age: middle, Hair color: None
            'senior': (None, 2, None, None),        # Age: senior, Hair color: None
            'ageothers': (None, 3, None, None),     # Age: others, Hair color: None
            
            # Hair Color
            'gray': (None, None, 1, None),    # Hair color: gray
            'blackh': (None, None, 2, None),   # Hair color: black
            'blond': (None, None, 3, None),   # Hair color: blond
            'hairothers': (None, None, 0, None),   # Hair color: others

            'noeyewear': (None, None, None, 1), # No eyewear
            'eyewear': (None, None, None, 2), # eyewear
            'otherseye': (None, None, None, 0) # other eyewear
        }

        # Check if the attribute is valid
        if attribute not in attribute_to_labels:
            raise ValueError(f"Attribute {attribute} is not recognized.")
        
        intersec_label, age_label, hair_label, eye_label = attribute_to_labels[attribute]

        with open(csv_file, newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=',')
            next(rows)  # Skip the header row
            for row in rows:
                img_path = row[0]
                mylabel = int(row[8])
                
                # Depending on the attribute, check the corresponding label
                if intersec_label is not None and int(row[7]) == intersec_label:
                    self.img.append(img_path)
                    self.label.append(mylabel)
                elif age_label is not None and int(row[5]) == age_label:
                    self.img.append(img_path)
                    self.label.append(mylabel)
                elif hair_label is not None and int(row[17]) == hair_label:
                    self.img.append(img_path)
                    self.label.append(mylabel)
                elif eye_label is not None and int(row[16]) == eye_label:
                    self.img.append(img_path)
                    self.label.append(mylabel)
      
    def __getitem__(self, index):
        path = self.img[index]
        img = np.array(Image.open(path))
        label = self.label[index]
        augmented = self.transform(image=img)
        img = augmented['image'] 

        data_dict = {
            'image': img,
            'label': label
        }

        return data_dict


    def __len__(self):
        return len(self.img)
