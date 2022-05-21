import torch
from torch.utils.data import Dataset
import albumentations as A
from transformers import ViTFeatureExtractor
from transformers import BertTokenizer, BigBirdTokenizer, XLNetTokenizer


def get_transforms(config):
    return A.Compose(
        [
            A.Resize(config.size, config.size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ]
    )


def get_tokenizer(config):
    if 'bigbird' in config.text_encoder_model:
        tokenizer = BigBirdTokenizer.from_pretrained(config.text_tokenizer)
    elif 'xlnet' in config.text_encoder_model:
        tokenizer = XLNetTokenizer.from_pretrained(config.text_tokenizer)
    else:
        tokenizer = BertTokenizer.from_pretrained(config.text_tokenizer)
    return tokenizer


class DatasetLoader(Dataset):
    def __init__(self, config, dataframe, mode):
        self.config = config
        self.mode = mode
        if mode == 'lime':
            self.image_filenames = [dataframe["image"],]
            self.text = [dataframe["text"],]
            self.labels = [dataframe["label"],]
        else:
            self.image_filenames = dataframe["image"].values
            self.text = list(dataframe["text"].values)
            self.labels = dataframe["label"].values

        tokenizer = get_tokenizer(config)
        self.encoded_text = tokenizer(self.text, padding=True, truncation=True, max_length=config.max_length, return_tensors='pt')
        if 'resnet' in config.image_model_name:
            self.transforms = get_transforms(config)
        else:
            self.transforms = ViTFeatureExtractor.from_pretrained(config.image_model_name)

    def set_text(self, idx):
        item = {
            key: values[idx].clone().detach()
            for key, values in self.encoded_text.items()
        }
        item['text'] = self.text[idx]
        item['label'] = self.labels[idx]
        item['id'] = idx
        return item

    def set_image(self, image):
        if 'resnet' in self.config.image_model_name:
            image = self.transforms(image=image)['image']
            return {'image': torch.as_tensor(image).reshape((3, 224, 224))}
        else:
            image = self.transforms(images=image, return_tensors='pt')
            image = image.convert_to_tensors(tensor_type='pt')['pixel_values']
            return {'image': image.reshape((3, 224, 224))}