import numpy as np 
import torch 
import os 
from glob import glob 
import monai 
from monai import transforms 
import nibabel as nib 
import random



class dataset():
    def __init__(self,file_path,subject_names, transform = None,infer=False):
        self.file_path = file_path
        self.subject_names = subject_names
        self.transform = transform
        self.Loader = transforms.LoadImage()
        self.infer = infer

    def __len__(self):
        return len(self.subject_names)
    
    def load_nii(self,subject_name):
        mri_path = self.file_path +"/" + subject_name + "/" + subject_name +"_mri_MNI152.nii.gz"
        # print(mri_path)
        pet_path = self.file_path +"/" + subject_name + "/" + subject_name +"_FDG_MNI152.nii.gz"

        mri_data = self.Loader(mri_path)[0].unsqueeze(0)
        pet_data = self.Loader(pet_path)[0].unsqueeze(0)

        return {
            "mri" : mri_data,
            "pet" : pet_data
        }

    def __getitem__(self,index):
        subject_name = self.subject_names[index]
        data = self.load_nii(subject_name)
        if self.transform is not None :
            data = self.transform(data)
        
        if self.infer:
            return data["mri"] , data["pet"],subject_name
        
        return data["mri"] , data["pet"]
    
