import os

from .utils import DatasetBase, Datum

template = ['a photo of a {}, a type of aircraft.']


class FGVCAircraft(DatasetBase):

    dataset_dir = 'fgvc_aircraft'

    def __init__(self, root, num_shots):

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')

        self.template = template

        classnames = []
        with open(os.path.join(self.dataset_dir, 'variants.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(classnames)}

        train = self.read_data(cname2lab, 'images_variant_train.txt')
        val = self.read_data(cname2lab, 'images_variant_val.txt')
        test = self.read_data(cname2lab, 'images_variant_test.txt')

        train_x = self.generate_fewshot_dataset(train, num_shots=num_shots)
        train_u = [item for item in train if item not in train_x]

        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)

    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []

        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                imname = line[0] + '.jpg'
                classname = ' '.join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=classname
                )
                items.append(item)

        return items
