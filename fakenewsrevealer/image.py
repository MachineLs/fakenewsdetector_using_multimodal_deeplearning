
from torch import nn
from transformers import ViTModel, ViTConfig


class ImageTransformerEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.pretrained:
            self.model = ViTModel.from_pretrained(config.image_model_name, output_attentions=False,
                                                                   output_hidden_states=True, return_dict=True)
        else:
            self.model = ViTModel(config=ViTConfig())

        for p in self.model.parameters():
            p.requires_grad = config.trainable

        self.target_token_idx = 0
        self.image_encoder_embedding = dict()

    def forward(self, ids, image):
        output = self.model(image)
        last_hidden_state = output.last_hidden_state[:, self.target_token_idx, :]

        # for i, id in enumerate(ids):
        #     id = int(id.detach().cpu().numpy())
        #     self.image_encoder_embedding[id] = last_hidden_state[i].detach().cpu().numpy()
        return last_hidden_state


class ImageResnetEncoder(nn.Module):

    def __init__(
            self, config
    ):
        super().__init__()
        self.model = timm.create_model(
            config.image_model_name, config.pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = config.trainable

    def forward(self, ids, image):
        return self.model(image)