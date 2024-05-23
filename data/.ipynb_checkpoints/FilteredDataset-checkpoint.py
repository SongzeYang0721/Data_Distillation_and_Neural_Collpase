import torch
from torch.utils.data import Dataset

class FilteredDataset(Dataset):
    def __init__(self, dataset, classes_to_include):
        self.dataset = dataset
        self.classes_to_include = classes_to_include
        self.indices = self._get_filtered_indices()

    def _get_filtered_indices(self):
        filtered_indices = []
        for idx, (_, label) in enumerate(self.dataset):
            if label in self.classes_to_include:
                filtered_indices.append(idx)
        return filtered_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image, label = self.dataset[actual_idx]
        return image, label