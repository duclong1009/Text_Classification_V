import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import BertDataset
from models import DecoderModel
from tokenizer import VnCoreTokenizer
from trainer import eval_fn
from utils import load_model


def main(arg):
    vncore_tokenizer = VnCoreTokenizer(arg.vncore_tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(arg.bert_tokenizer, use_fast=False)
    bert_model = arg.bert_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderModel(bert_model, arg.n_class, 0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arg.lr)
    load_model(model, optimizer, torch.load(arg.path_checkpoint + arg.name_model))
    test_df = pd.read_excel(arg.fig_root + "/news.xlsx")
    test_df, _ = train_test_split(test_df, train_size=0.005)
    test_dataset = BertDataset(test_df, tokenizer, arg.max_len, vncore_tokenizer)
    test_dataloder = DataLoader(test_dataset, arg.batch_size, shuffle=False)
    output_test, target_test = eval_fn(test_dataloder, model, device)
    test_acc = sum(np.array(output_test) == np.array(target_test)) / len(target_test)
    print("Accuracy : ", test_acc)
    if arg.print_pred:
        print(
            np.concatenate(
                (np.expand_dims(output_test, -1), np.expand_dims(output_test, -1)), -1
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--bert_model", type=str, default="vinai/phobert-base")
    parser.add_argument("--bert_tokenizer", type=str, default="vinai/phobert-base")
    parser.add_argument(
        "--vncore_tokenizer", type=str, default="./vncorenlp/VnCoreNLP-1.1.1.jar"
    )
    parser.add_argument("--path_checkpoint", type=str, default="./checkpoint/")
    parser.add_argument("--name_model", type=str, default="510_first_token")
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--n_class", type=int, default=4)
    parser.add_argument("--fig_root", type=str, default="./data")
    parser.add_argument("--print_pred", type=bool, default=False)
    args = parser.parse_args()
    main(args)
