from torch import nn
from transformers import BertModel, BertConfig, BertTokenizer, \
    BigBirdModel, BigBirdConfig, BigBirdTokenizer, \
    XLNetModel, XLNetConfig, XLNetTokenizer


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = get_text_model(config)
        for p in self.model.parameters():
            p.requires_grad = self.config.trainable
        self.target_token_idx = 0
        self.text_encoder_embedding = dict()

    def forward(self, ids, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state[:, self.target_token_idx, :]

        # for i, id in enumerate(ids):
        #     id = int(id.detach().cpu().numpy())
        #     self.text_encoder_embedding[id] = last_hidden_state[i].detach().cpu().numpy()
        return last_hidden_state


def get_text_model(config):
    if 'bigbird' in config.text_encoder_model:
        if config.pretrained:
            model = BigBirdModel.from_pretrained(config.text_encoder_model, block_size=16, num_random_blocks=2)
        else:
            model = BigBirdModel(config=BigBirdConfig())
    elif 'xlnet' in config.text_encoder_model:
        if config.pretrained:
            model = XLNetModel.from_pretrained(config.text_encoder_model, output_attentions=False,
                                              output_hidden_states=True, return_dict=True)
        else:
            model = XLNetModel(config=XLNetConfig())
    else:
        if config.pretrained:
            model = BertModel.from_pretrained(config.text_encoder_model, output_attentions=False,
                                              output_hidden_states=True, return_dict=True)
        else:
            model = BertModel(config=BertConfig())
    return model

