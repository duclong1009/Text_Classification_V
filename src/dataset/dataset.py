import math
import re

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoTokenizer

from vncorenlp import VnCoreNLP


class BertDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_len: int,
        corevn_tokenizer,
    ):
        super(BertDataset, self).__init__()
        """
      Args:
        df(pd.DataFrame): DataFrame for all arguments
        tokenizer : Pretrained PhoBert
    """
        self.corevn_tokenizer = corevn_tokenizer
        self.max_len = max_len
        self.df = df.copy()
        self.content = df["content"].values
        self.label = df["label"].values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        content = self.content[idx]
        content = self.corevn_tokenizer.tokenize(content)
        label = self.label[idx]
        (
            content_input_ids,
            content_attention_mask,
            content_token_type_ids,
        ) = self._tokenize(text=content, max_len=self.max_len)
        sample = {
            "content_input_ids": torch.tensor(content_input_ids, dtype=torch.long),
            "content_attention_mask": torch.tensor(
                content_attention_mask, dtype=torch.long
            ),
            "content_token_type_ids": torch.tensor(
                content_token_type_ids, dtype=torch.long
            ),
            "label": torch.tensor(label, dtype=torch.float),
        }
        return sample

    def _tokenize(self, text: str, max_len: int):
        inputs = self.tokenizer.encode_plus(
            text,
            # add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return input_ids, attention_mask, token_type_ids


def load_data(df: pd.DataFrame):
    x = df["content"].to_list()
    y = df["label"].to_list()
    return x, y


def load_segments(df, vncore_tokenizer, size_segment=200, size_shift=50):
    x_full, y_full = load_data(df)

    def get_segments(sentences):
        list_segment = []
        length_ = size_segment - size_shift
        sentences = vncore_tokenizer.tokenize(sentences)
        token = sentences.split(" ")
        n_tokens = len(token)
        n_segment = math.ceil(n_tokens / length_)

        if n_segment > 1:
            for i in range(0, n_tokens, length_):
                j = min(i + size_segment, n_tokens)
                list_segment.append(" ".join(token[i:j]))
        else:
            list_segment.append(sentences)

        return list_segment, len(list_segment)

    def get_segments_from_section(sentences):
        list_segments = []
        list_num_segments = []
        for sentence in sentences:
            ls, ns = get_segments(sentence)
            list_segments += ls
            list_num_segments.append(ns)
        return list_segments, list_num_segments

    x, num_segments = get_segments_from_section(x_full)

    return x, y_full, num_segments


def generate_dataset(
    X,
    Y,
    num_segments,
    tokenizer,
    pad_to_max_length=True,
    add_special_tokens=True,
    size_segment=200,
    return_attention_mask=True,
):
    """ """
    tokens = tokenizer.batch_encode_plus(
        X,
        pad_to_max_length=pad_to_max_length,
        add_special_tokens=add_special_tokens,
        max_length=size_segment,
        return_attention_mask=return_attention_mask,  # 0: padded tokens, 1: not padded tokens; taking into account the sequence length
        return_tensors="pt",
    )
    num_sentences = len(Y)
    max_segments = max(num_segments)
    input_ids = torch.zeros(
        (num_sentences, max_segments, size_segment), dtype=tokens["input_ids"].dtype
    )
    attention_mask = torch.zeros(
        (num_sentences, max_segments, size_segment),
        dtype=tokens["attention_mask"].dtype,
    )
    token_type_ids = torch.zeros(
        (num_sentences, max_segments, size_segment),
        dtype=tokens["token_type_ids"].dtype,
    )
    # pad_token = 0
    pos_segment = 0
    for idx_segment, n_segments in enumerate(num_segments):
        for n in range(n_segments):
            input_ids[idx_segment, n] = tokens["input_ids"][pos_segment]
            attention_mask[idx_segment, n] = tokens["attention_mask"][pos_segment]
            token_type_ids[idx_segment, n] = tokens["token_type_ids"][pos_segment]
            pos_segment += 1
    dataset = TensorDataset(
        input_ids,
        attention_mask,
        token_type_ids,
        torch.tensor(num_segments),
        torch.tensor(Y),
    )
    return dataset


class VnCoreTokenizer:
    def __init__(self, path="vncorenlp/VnCoreNLP-1.1.1.jar"):
        self.rdrsegmenter = VnCoreNLP(path, annotators="wseg", max_heap_size="-Xmx500m")

    def tokenize(self, text: str, return_sentences=False) -> str:
        sentences = self.rdrsegmenter.tokenize(text)
        if return_sentences:
            return [" ".join(sentence) for sentence in sentences]
        output = ""

        for sentence in sentences:
            output += " ".join(sentence) + " "
        return self._strip_white_space(output)

    def _strip_white_space(self, text):
        text = re.sub("\n+", "\n", text).strip()
        text = re.sub(" +", " ", text).strip()
        return text


import torch.nn as nn
from transformers import AutoConfig, AutoModel


class GRU_BERT(nn.Module):
    def __init__(self, bert_model, n_classes, dropout, hid_gru_dim):
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

    def forward(self, x):
        content_input_ids, content_attention_mask, content_token_type_ids = (
            x["input_ids"],
            x["attention_mask"],
            x["token_type_ids"],
        )
        print(content_input_ids.shape)
        n_sentences, max_segments, segment_length = content_input_ids.size()
        total_segment = n_sentences * max_segments

        content_input_ids = content_input_ids.view(total_segment, segment_length)
        content_attention_mask = content_attention_mask.view(
            total_segment, segment_length
        )
        content_token_type_ids = content_token_type_ids.view(
            total_segment, segment_length
        )
        print(content_input_ids.shape)
        out = self.bert(
            content_input_ids,
            attention_mask=content_attention_mask,
            token_type_ids=content_token_type_ids,
        )
        output = out.last_hidden_state
        output = self.bert_drop(output)
        output = output[:, 0, :]
        print(output.shape)
        output = output.view(n_sentences, max_segments, self.size_token_embed)
        print(output.shape)
        output, hidden_state = self.gru(output)
        print(output.shape)
        output = output[:, -1, :]
        return self.softmax(self.fc(output))


if __name__ == "__main__":
    # from src.bert.models import GRU_BERT

    vncore_tokenizer = VnCoreTokenizer("./vncorenlp/VnCoreNLP-1.1.1.jar")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    bert_model = "vinai/phobert-base"
    val_df = pd.read_excel("./data/train.xlsx")
    train_df, val_df = train_test_split(
        val_df, train_size=0.005, test_size=0.2, stratify=val_df["label"]
    )
    x_train, y_train, num_segments_train = load_segments(train_df, vncore_tokenizer)
    dataset = generate_dataset(x_train, y_train, num_segments_train, tokenizer)
    train_dataloader = DataLoader(dataset, batch_size=4)
    model = GRU_BERT(bert_model, 4, 0.2, 200)
    for i in train_dataloader:
        input = {"input_ids": i[0], "attention_mask": i[1], "token_type_ids": i[2]}
        output = model(input)
        print(output.shape)
        break


# import torch

# if __name__ == "__main__":
#     from sklearn.model_selection import train_test_split

#     from trainer import eval_fn

#     vncore_tokenizer = VnCoreTokenizer("./vncorenlp/VnCoreNLP-1.1.1.jar")
#     tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
#     bert_model = "vinai/phobert-base"
#     val_df = pd.read_excel("./data/train.xlsx")
#     train_df, val_df = train_test_split(val_df, test_size=0.2, stratify=val_df["label"])
#     val_dataset = BertDataset(val_df, tokenizer, 64, vncore_tokenizer)
#     val_dataloader = DataLoader(val_dataset, batch_size=1)
#     model = DecoderModel(bert_model, 4, 0.3)
#     output, target = eval_fn(val_dataloader, model, torch.device("cpu"))
#     for input in val_dataloader:
#         output = model(
#             content_input_ids=input["content_input_ids"],
#             content_attention_mask=input["content_attention_mask"],
#             content_token_type_ids=input["content_token_type_ids"],
#         )
#         print(output.shape)
#         break
