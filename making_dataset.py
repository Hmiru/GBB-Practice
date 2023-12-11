import random
from preprocess import *
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import yaml
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
with open('config.yaml') as f:
    file = yaml.full_load(f)

DATASET_LOCATION = file["dataset_path"]
data_set = ImageFolder(DATASET_LOCATION,
                       transform=make_composed_transform_with_size(224),
                       target_transform=make_one_hot_transform_with_class_num(9)
                       )

data_loader = DataLoader(data_set, batch_size=32, shuffle=True)
label_to_class, class_to_label = data_set.find_classes(DATASET_LOCATION)
def get_dataloader(batch_size, train, valid):

    total_size=len(data_set)
    indices = list(range(total_size))
    random.shuffle(indices)

    train_split = int(train * total_size)
    valid_split = int(valid * total_size)

    train_indices = indices[:train_split]
    valid_indices = indices[train_split:train_split + valid_split]
    test_indices = indices[train_split + valid_split:]

    train_set = Subset(data_set, train_indices)
    valid_set = Subset(data_set, valid_indices)
    test_set = Subset(data_set, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

if __name__=="__main__":
    pass