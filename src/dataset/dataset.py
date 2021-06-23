import math
import random
import re

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from vncorenlp import VnCoreNLP


class TailTokenDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_len: int,
        corevn_tokenizer,
    ):
        super(TailTokenDataset, self).__init__()
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
        text = self.corevn_tokenizer.tokenize(text)
        list_text = text.split(" ")
        if len(list_text) > max_len:
            list_text = list_text[-max_len:]
        text = " ".join(list_text)
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return input_ids, attention_mask, token_type_ids


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
        text = self.corevn_tokenizer.tokenize(text)
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return input_ids, attention_mask, token_type_ids


class RandomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_len: int,
        corevn_tokenizer,
    ):
        super(RandomDataset, self).__init__()
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
        text = self.corevn_tokenizer.tokenize(text)
        list_text = text.split(" ")
        length = len(list_text)
        index = random.randint(0, length)
        if (length - index) > max_len:
            text = " ".join(list_text[index : index + max_len])
        else:
            text = " ".join(list_text[index:])
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return input_ids, attention_mask, token_type_ids


class VnCoreTokenizer:
    def __init__(self, path="vncorenlp/VnCoreNLP-1.1.1.jar"):
        self.rdrsegmenter = VnCoreNLP(path, annotators="wseg", max_heap_size="-Xmx500m")

    def tokenize(self, text: str, return_sentences=False) -> str:
        sentences = self.rdrsegmenter.tokenize(text)
        # print(sentence)
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


class GB_Dataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vncore_tokenizer,
        tokenizer: AutoTokenizer,
        max_segments=40,
        size_segment=200,
        size_shift=50,
    ):
        super().__init__()
        self.df = df
        self.vncore_tokenizer = vncore_tokenizer
        self.tokenizer = tokenizer
        self.size_segment = size_segment
        self.size_shift = size_shift
        self.max_segments = max_segments
        self.content = df["content"].values
        self.label = df["label"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        content = self.content[idx]
        label = self.label[idx]
        content = self.vncore_tokenizer.tokenize(content)
        list_segment, n_segment = self.get_segments(content)
        (
            content_input_ids,
            content_attention_mask,
            content_token_type_ids,
        ) = self.generate_dataset(list_segment, n_segment)
        sample = {
            "content_input_ids": content_input_ids,
            "content_attention_mask": content_attention_mask,
            "content_token_type_ids": content_token_type_ids,
            "label": label,
        }
        return sample

    def get_segments(self, sentences):
        list_segment = []
        length_ = self.size_segment - self.size_shift
        token = sentences.split(" ")
        n_tokens = len(token)
        n_segment = math.ceil(n_tokens / length_)
        if n_segment > self.max_segments:
            n_segment = self.max_segments
        if n_segment > 1:
            for i in range(n_segment):
                j = min(i + self.size_segment, n_tokens)
                list_segment.append(" ".join(token[i:j]))
        else:
            list_segment.append(sentences)

        return list_segment, len(list_segment)

    def generate_dataset(self, X, num_segments):
        """ """
        tokens = self.tokenizer.batch_encode_plus(
            X,
            padding="max_length",
            add_special_tokens=True,
            max_length=self.size_segment,
            return_attention_mask=True,  # 0: padded tokens, 1: not padded tokens; taking into account the sequence length
            return_tensors="pt",
            truncation=True,
        )
        input_ids = torch.zeros(
            (self.max_segments, self.size_segment), dtype=tokens["input_ids"].dtype
        )
        attention_mask = torch.zeros(
            (self.max_segments, self.size_segment),
            dtype=tokens["attention_mask"].dtype,
        )
        token_type_ids = torch.zeros(
            (self.max_segments, self.size_segment),
            dtype=tokens["token_type_ids"].dtype,
        )
        for n in range(num_segments):
            input_ids[n, :] = tokens["input_ids"][n]
            attention_mask[n, :] = tokens["attention_mask"][n]
            token_type_ids[n:] = tokens["token_type_ids"][n]
        return input_ids, attention_mask, token_type_ids


if __name__ == "__main__":
    path = "vncorenlp/VnCoreNLP-1.1.1.jar"
    vncore_tokenizer = VnCoreNLP(path, annotators="wseg", max_heap_size="-Xmx500m")
    t = "Tôi tên là Nguyễn Đức Long. Tôi là sinh viên năm ba."
    print(vncore_tokenizer.tokenize(t))

# if __name__ == "__main__":
#     # from src.bert.models import GRU_BERT

#     vncore_tokenizer = VnCoreTokenizer("./vncorenlp/VnCoreNLP-1.1.1.jar")
#     tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
#     bert_model = "vinai/phobert-base"
#     val_df = pd.read_excel("./data/train.xlsx")
#     train_df, val_df = train_test_split(
#         val_df, train_size=0.005, test_size=0.2, stratify=val_df["label"]
#     )
#     x_train, y_train, num_segments_train = load_segments(train_df, vncore_tokenizer)
#     dataset = generate_dataset(x_train, y_train, num_segments_train, tokenizer)
#     train_dataloader = DataLoader(dataset, batch_size=3)
#     model = GRU_BERT(bert_model, 4, 0.2, 200)
#     for i in train_dataloader:
#         # i shape 1
#         print(type(i))
#         print(len(i))
#         input = {"input_ids": i[0], "attention_mask": i[1], "token_type_ids": i[2]}
#         print(i[0].shape)
#         # output = model(input)
#         # print(output.shape)
#         break


import torch

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    vncore_tokenizer = VnCoreTokenizer("./vncorenlp/VnCoreNLP-1.1.1.jar")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    bert_model = "vinai/phobert-base"
    val_df = pd.read_excel("./data/train.xlsx")
    train_df, val_df = train_test_split(
        val_df, test_size=0.006, stratify=val_df["label"]
    )
    val_dataset = GB_Dataset(val_df, vncore_tokenizer, tokenizer, 40, 200, 50)
    val_dataloader = DataLoader(val_dataset, batch_size=3)
    # for i in val_dataloader:
    #     print(i["label"])
    #     break
    # model = DecoderModel(bert_model, 4, 0.3)
    # output, target = eval_fn(val_dataloader, model, torch.device("cpu"))
    # for input in val_dataloader:
    #     output = model(
    #         content_input_ids=input["content_input_ids"],
    #         content_attention_mask=input["content_attention_mask"],
    #         content_token_type_ids=input["content_token_type_ids"],
    #     )
    #     print(output.shape)
    #     break
