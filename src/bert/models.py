import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class DecoderModel(nn.Module):
    def __init__(self, bert_model="vinai/phobert-base", n_classes=4, dropout=0.3):
        super().__init__()
        config = AutoConfig.from_pretrained(
            bert_model,
            from_tf=False,
            output_hidden_states=True,
        )
        self.bert = AutoModel.from_pretrained(bert_model, config=config)
        self.bert_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(config.hidden_size * 4, n_classes)
        self.softmax = nn.Softmax()
        for param in self.bert.parameters():
            param.requires_grad = False
        self.weight_init(self.fc)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

    def forward(
        self,
        content_input_ids=None,
        content_attention_mask=None,
        content_token_type_ids=None,
    ):
        output = self.bert(
            content_input_ids,
            attention_mask=content_attention_mask,
            token_type_ids=content_token_type_ids,
        )
        # output = out.last_hidden_state
        # output = torch.mean(output, dim=1)
        output = torch.cat([output[2][-i][:, 0, :] for i in range(4)], axis=-1)
        out = self.bert_drop(output)
        out = self.fc(out)
        return self.softmax(out)


class GRU_BERT(nn.Module):
    def __init__(
        self, bert_model="vinai/phobert-base", n_classes=4, dropout=0.3, hid_gru_dim=128
    ):
        super(GRU_BERT, self).__init__()
        config = AutoConfig.from_pretrained(
            bert_model,
            from_tf=False,
            # output_hidden_states=True,
        )
        self.size_token_embed = config.hidden_size
        self.bert = AutoModel.from_pretrained(bert_model, config=config)
        self.bert_drop = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=hid_gru_dim,
            bidirectional=True,
            batch_first=True,
        )

        self.fc = nn.Linear(in_features=hid_gru_dim * 2, out_features=n_classes)
        self.softmax = nn.Softmax(dim=-1)
        self.weight_init(self.fc)
        for param in self.bert.parameters():
            param.requires_grad = False

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

    def forward(self, x=None):
        content_input_ids, content_attention_mask, content_token_type_ids = (
            x["input_ids"],
            x["attention_mask"],
            x["token_type_ids"],
        )
        n_sentences, max_segments, segment_length = content_input_ids.size()
        total_segment = n_sentences * max_segments

        content_input_ids = content_input_ids.view(total_segment, segment_length)
        content_attention_mask = content_attention_mask.view(
            total_segment, segment_length
        )
        content_token_type_ids = content_token_type_ids.view(
            total_segment, segment_length
        )

        out = self.bert(
            content_input_ids,
            attention_mask=content_attention_mask,
            token_type_ids=content_token_type_ids,
        )
        output = out.last_hidden_state
        output = self.bert_drop(output)
        output = output[:, 0, :]
        output = output.view(n_sentences, max_segments, self.size_token_embed)
        output, hidden_state = self.gru(output)
        output = output[:, -1, :]
        return self.softmax(self.fc(output))


class FC_BERT(nn.Module):
    def __init__(
        self, bert_model="vinai/phobert-base", n_classes=4, dropout=0.3, n_segments=10
    ):
        super(FC_BERT, self).__init__()
        config = AutoConfig.from_pretrained(
            bert_model,
            from_tf=False,
            # output_hidden_states=True,
        )
        self.size_token_embed = config.hidden_size
        self.bert = AutoModel.from_pretrained(bert_model, config=config)
        self.bert_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(
            in_features=config.hidden_size * n_segments, out_features=n_classes
        )
        self.softmax = nn.Softmax(dim=-1)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.weight_init(self.fc)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

    def forward(self, x=None):
        content_input_ids, content_attention_mask, content_token_type_ids = (
            x["input_ids"],
            x["attention_mask"],
            x["token_type_ids"],
        )
        n_sentences, max_segments, segment_length = content_input_ids.size()
        total_segment = n_sentences * max_segments

        content_input_ids = content_input_ids.view(total_segment, segment_length)
        content_attention_mask = content_attention_mask.view(
            total_segment, segment_length
        )
        content_token_type_ids = content_token_type_ids.view(
            total_segment, segment_length
        )

        out = self.bert(
            content_input_ids,
            attention_mask=content_attention_mask,
            token_type_ids=content_token_type_ids,
        )
        output = out.last_hidden_state
        output = self.bert_drop(output)
        output = output[:, 0, :]
        output = output.view(n_sentences, max_segments, self.size_token_embed)
        output = output.reshape(n_sentences, -1)
        # output = torch.mean(output, dim=1)

        # output, hidden_state = self.gru(output)
        # output = output[:, -1, :]
        return self.softmax(self.fc(output))
