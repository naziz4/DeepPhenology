# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 21:42:51 2021

@author: Naeem
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:44:03 2021

@author: Naeem
"""
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import models, transforms
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image

#------------------------------------------------------------------------------
class CustomDataSet(Dataset):
    def __init__(self, csv_file, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.dataframe = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.dataframe.iloc[idx, 15]
        image = Image.open(img_path).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image
#------------------------------------------------------------------------------
#def locations
input_file = '/ihome/skuebbing/nza4/ml/csv/prediction_data.csv'
img_list = pd.read_csv(input_file)
img_dir = '/ihome/skuebbing/nza4/ml/prediction_data'

#Dataloader
transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

dataset = CustomDataSet(csv_file=input_file, root_dir=img_dir,
                                 transform = transforms)
testloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=0)

#Run predictions
model = models.resnet18(pretrained = True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('/ihome/skuebbing/nza4/ml/models/2_stage_model.pt'))
model.eval()

classes = ('Flowers', 'No_Flowers')
class_probs = []
class_preds = []
with torch.no_grad():
    for data in testloader:
        images = data
        output = model(images)
        class_probs_batch = [F.softmax(el, dim=0) for el in output]
        _, class_preds_batch = torch.max(output, 1)

        class_probs.append(class_probs_batch)
        class_preds.append(class_preds_batch)

test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_preds = torch.cat(class_preds)

#Get model output
preds = test_preds.numpy()
probs = test_probs.numpy()
model_output = np.column_stack((preds, probs))
model_output = pd.DataFrame(model_output, columns = ['Label', 'f_prob', 'nf_prob'])
model_output['imagepaths'] = list(img_list['imagepaths'])
model_output.to_csv('/ihome/skuebbing/nza4/ml/csv/model_predictions_2.csv')