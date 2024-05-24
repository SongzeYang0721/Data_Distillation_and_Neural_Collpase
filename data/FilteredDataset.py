import torch
from torch.utils.data import Dataset
class FilteredDataset(Dataset):
    def __init__(self, dataset, classes_to_include):
        self.dataset = dataset
        self.classes_to_include = classes_to_include
        self.label_map = {original_label: new_label for new_label, original_label in enumerate(classes_to_include)}
        self.filtered_indices = self._get_filtered_indices()

    def _get_filtered_indices(self):
        return [idx for idx, (_, label) in enumerate(self.dataset) if label in self.classes_to_include]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        actual_idx = self.filtered_indices[idx]
        image, label = self.dataset[actual_idx]
        remapped_label = self.label_map[label]
        return image, remapped_label