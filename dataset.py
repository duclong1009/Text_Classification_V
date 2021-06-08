import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModel
from tokenizer import VnCoreTokenizer


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
        print(content)
        content = self.corevn_tokenizer.tokenize(content)
        print("----------------------------------")
        print(content)
        print("-----------------------------------")
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


# if __name__ == "__main__":
#     vncore_tokenizer = VnCoreTokenizer("./vncorenlp/VnCoreNLP-1.1.1.jar")
#     tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
#     bert_model = "vinai/phobert-base"
#     config = AutoConfig.from_pretrained(bert_model, from_tf=False)
#     bert = AutoModel.from_pretrained(bert_model, config)
#     val_df = pd.read_excel("./data/train.xlsx")
#     val_dataset = BertDataset(val_df, tokenizer, 64, vncore_tokenizer)
#     print(val_dataset[1])