
import cv2
import pickle
from PIL import Image
from numpy import asarray

from data.data_loader import DatasetLoader


class TwitterDatasetLoader(DatasetLoader):
    def __getitem__(self, idx):
        item = self.set_text(idx)
        item.update(self.set_img(idx))
        return item

    def set_img(self, idx):
        try:
            if self.mode == 'train':
                image = cv2.imread(f"{self.config.train_image_path}/{self.image_filenames[idx]}")
            elif self.mode == 'validation':
                image = cv2.imread(f"{self.config.validation_image_path}/{self.image_filenames[idx]}")
            else:
                image = cv2.imread(f"{self.config.test_image_path}/{self.image_filenames[idx]}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            image = Image.open(f"{self.config.train_image_path}/{self.image_filenames[idx]}").convert('RGB')
            image = asarray(image)
        return self.set_image(image)

    def __len__(self):
        return len(self.text)

