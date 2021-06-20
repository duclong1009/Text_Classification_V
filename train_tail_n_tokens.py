import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.bert.models import DecoderModel
from src.bert.trainer import eval_fn, train_fn
from src.dataset.dataset import TailTokenDataset, VnCoreTokenizer
from src.utils.utils import *


def main(arg):
    seed_all(arg.seed)
    vncore_tokenizer = VnCoreTokenizer(arg.vncore_tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(arg.bert_tokenizer, use_fast=False)
    bert_model = arg.bert_model
    df = pd.read_excel(arg.root_path + "data/news.xlsx")
    train_df, val_df = train_test_split(
        df, test_size=arg.test_size, stratify=df["label"]
    )
    if arg.upsampling:
        a = train_df[train_df["label"] != 3]
        temp = pd.concat((a, a, a, train_df[train_df["label"] == 3])).reset_index()
        train_df = temp[["content", "label"]]
    train_dataset = TailTokenDataset(train_df, tokenizer, arg.max_len, vncore_tokenizer)
    val_dataset = TailTokenDataset(val_df, tokenizer, arg.max_len, vncore_tokenizer)
    train_dataloader = DataLoader(
        train_dataset, batch_size=arg.batch_size, shuffle=True
    )
    val_dataloder = DataLoader(val_dataset, batch_size=arg.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderModel(bert_model, arg.n_class, 0.3).to(device)
    params = [params for params in model.parameters() if params.requires_grad == True]
    print(count_parameters(model))
    optimizer = torch.optim.AdamW(params, lr=arg.lr)
    CE_Loss = nn.CrossEntropyLoss()
    path_save = arg.path_save
    es = EarlyStopping(3, path=(path_save))

    for i in range(arg.epochs):
        loss = train_fn(train_dataloader, model, optimizer, CE_Loss, device)
        output, target = eval_fn(val_dataloder, model, device)
        accuracy = sum(np.array(output) == np.array(target)) / len(target)
        print(
            "epochs {} / {}  train_loss : {}  val_acc : {}".format(
                i + 1, arg.epochs, loss, accuracy
            )
        )
        es(accuracy, model)

    load_model(model, torch.load(path_save))
    test_df = pd.read_excel(arg.root_path + "data/news.xlsx")
    test_dataset = BertDataset(test_df, tokenizer, arg.max_len, vncore_tokenizer)
    test_dataloder = DataLoader(test_dataset, arg.batch_size, shuffle=False)
    output_test, target_test = eval_fn(test_dataloder, model, device)
    test_acc = sum(np.array(output_test) == np.array(target_test)) / len(target_test)
    print("Accuracy test: ", test_acc)


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
    parser.add_argument("--path_save", type=str, default="./200_first_token")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--n_class", type=int, default=4)
    parser.add_argument("--root_path", type=str, default="./")
    parser.add_argument("--upsampling", type=bool, default=False)
    args = parser.parse_args()
    main(args)
