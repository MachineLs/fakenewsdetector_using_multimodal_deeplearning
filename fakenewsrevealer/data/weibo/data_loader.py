
import pickle
import cv2
from PIL import Image
from numpy import asarray

from data.data_loader import DatasetLoader



class WeiboDatasetLoader(DatasetLoader):
    def __getitem__(self, idx):
        item = self.set_text(idx)
        item.update(self.set_img(idx))
        return item

    def set_img(self, idx):
        if self.labels[idx] == 1:
            try:
                image = cv2.imread(f"{self.config.rumor_image_path}/{self.image_filenames[idx]}")
            except:
                image = Image.open(f"{self.config.rumor_image_path}/{self.image_filenames[idx]}").convert('RGB')
                image = asarray(image)
        else:
            try:
                image = cv2.imread(f"{self.config.nonrumor_image_path}/{self.image_filenames[idx]}")
            except:
                image = Image.open(f"{self.config.nonrumor_image_path}/{self.image_filenames[idx]}").convert('RGB')
                image = asarray(image)
        return self.set_image(image)

    def __len__(self):
        return len(self.text)

