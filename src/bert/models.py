import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class DecoderModel(nn.Module):
    def __init__(self, bert_model, n_classes, dropout):
        super().__init__()
        config = AutoConfig.from_pretrained(
            bert_model,
            from_tf=False,
            # output_hidden_states=True,
        )
        self.bert = AutoModel.from_pretrained(bert_model, config=config)
        self.bert_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(config.hidden_size, n_classes)
        self.softmax = nn.Softmax()

    def forward(
        self, content_input_ids, content_attention_mask, content_token_type_ids
    ):
        out = self.bert(
            content_input_ids,
            attention_mask=content_attention_mask,
            token_type_ids=content_token_type_ids,
        )
        output = out.last_hidden_state
        output = torch.mean(output, dim=1)
        out = self.bert_drop(output)
        out = self.fc(out)
        return self.softmax(out)
