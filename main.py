import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import BertDataset
from models import DecoderModel
from tokenizer import VnCoreTokenizer
from trainer import eval_fn, train_fn
from utils import EarlyStopping, seed_all


def main(arg):
    seed_all(arg.seed)
    vncore_tokenizer = VnCoreTokenizer(arg.vncore_tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(arg.bert_tokenizer, use_fast=False)
    bert_model = arg.bert_model
    df = pd.read_excel("./data/train.xlsx")
    train_df, val_df = train_test_split(
        df, test_size=0.001, train_size=0.005, stratify=df["label"]
    )
    train_dataset = BertDataset(train_df, tokenizer, arg.max_len, vncore_tokenizer)
    val_dataset = BertDataset(val_df, tokenizer, arg.max_len, vncore_tokenizer)
    train_dataloader = DataLoader(
        train_dataset, batch_size=arg.batch_size, shuffle=True
    )

    val_dataloder = DataLoader(val_dataset, batch_size=arg.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderModel(bert_model, arg.n_class, 0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arg.lr)
    CE_Loss = nn.CrossEntropyLoss()
    es = EarlyStopping(3, path=arg.path_checkpoint)
    for i in range(arg.epochs):
        loss = train_fn(train_dataloader, model, optimizer, CE_Loss, device)
        output, target = eval_fn(val_dataloder, model, device)
        accuracy = sum(np.array(output) == np.array(target)) / len(target)
        print(
            "epochs {} / {}  train_loss : {}  val_acc : {}".format(
                i + 1, arg.epochs, loss, accuracy
            )
        )
        es(accuracy, model, optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--bert_model", type=str, default="vinai/phobert-base")
    parser.add_argument("--bert_tokenizer", type=str, default="vinai/phobert-base")
    parser.add_argument(
        "--vncore_tokenizer", type=str, default="./vncorenlp/VnCoreNLP-1.1.1.jar"
    )
    parser.add_argument("--path_checkpoint", type=str, default="./checkpoint")
    parser.add_argument("--test_size", type=int, default=0.2)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--n_class", type=int, default=4)
    parser.add_argument("--fig_root", type=str, default="./data")
    args = parser.parse_args()
    main(args)
