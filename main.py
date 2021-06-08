from dataset import BertDataset
from models import DecoderModel
from trainer import train_fn
from utils import seed_all
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import argparse
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tokenizer import VnCoreTokenizer


def main(arg):
    seed_all(arg.seed)
    df = pd.read_excel(arg.fig_root + "/train.xlsx")
    print(df.shape)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"])
    tokenizer = AutoTokenizer.from_pretrained(arg.bert_tokenizer, use_fast=False)
    vncore_tokenizer = VnCoreTokenizer(arg.vncore_tokenizer)
    train_dataset = BertDataset(train_df, tokenizer, 64, vncore_tokenizer)
    val_dataset = BertDataset(val_df, tokenizer, 64, vncore_tokenizer)
    train_dataloader = DataLoader(
        train_dataset, batch_size=arg.batch_size, num_workers=2, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=arg.batch_size, num_workers=2, shuffle=False
    )
    print(train_dataset.__len__())
    model = DecoderModel(arg.bert_model, arg.n_class, 0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arg.lr)
    CE_Loss = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    for i in range(arg.epochs):
        print(i)
        loss = train_fn(train_dataloader, model, optimizer, CE_Loss, device)
    print("epochs {} / 5  train_loss {}: ".format(i + 1, loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--bert_model", type=str, default="vinai/phobert-base")
    parser.add_argument("--bert_tokenizer", type=str, default="vinai/phobert-base")
    parser.add_argument(
        "--vncore_tokenizer", type=str, default="./vncorenlp/VnCoreNLP-1.1.1.jar"
    )
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--n_class", type=int, default=4)
    parser.add_argument("--fig_root", type=str, default="./data")
    args = parser.parse_args()
    main(args)
