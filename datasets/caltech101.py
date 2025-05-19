import os

from .oxford_pets import OxfordPets
from .utils import DatasetBase

template = ['a photo of a {}.']


class Caltech101(DatasetBase):

    dataset_dir = 'caltech-101'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '101_ObjectCategories')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_Caltech101.json')

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train_x = self.generate_fewshot_dataset(train, num_shots=num_shots)
        train_u = [item for item in train if item not in train_x]

        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)
