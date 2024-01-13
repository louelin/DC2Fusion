import csv 
import numpy as np 
import os 
from glob import glob
import random

if __name__ == "__main__":
    files = glob("preproce_data2/*")

    subject_names =  []

    for path in files :
        _ , subject_name = os.path.split(path)
        subject_names.append(subject_name)

    n = len(subject_names)
    train_index = random.sample(range(n),int(n * 0.9))
    
    train_subjects = [subject_names[i] for i in range(n) if i in train_index]
    val_subjects = [subject_names[i] for i in range(n) if i not in train_index]

    print(train_subjects)
    print(val_subjects)

    with open("/home/fanxx/fxx/Fusion/Fusion_MRI_PET/data/train.csv" , "w") as f:
        writer = csv.writer(f)
        writer.writerow(train_subjects)

    with open("/home/fanxx/fxx/Fusion/Fusion_MRI_PET/data/test.csv" , "w") as f:
        writer = csv.writer(f)
        writer.writerow(val_subjects)