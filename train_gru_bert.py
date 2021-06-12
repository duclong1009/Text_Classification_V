import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.bert.models import GRU_BERT
from src.bert.trainer import eval_gru_fn, train_gru_fn
from src.dataset.dataset import *
from src.utils.utils import EarlyStopping, seed_all


def main(arg):
    seed_all(arg.seed)
    vncore_tokenizer = VnCoreTokenizer(arg.vncore_tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(arg.bert_tokenizer, use_fast=False)
    df = pd.read_excel(arg.root_path + "data/train.xlsx")
    train_df, val_df = train_test_split(
        df, test_size=arg.test_size, stratify=df["label"]
    )
    x_train, y_train, num_segments_train = load_segments(
        train_df, vncore_tokenizer, arg.size_segment, arg.size_shift
    )
    x_val, y_val, num_segments_val = load_segments(
        val_df, vncore_tokenizer, arg.size_segment, arg.size_shift
    )
    train_dataset = generate_dataset(x_train, y_train, num_segments_train, tokenizer)
    val_dataset = generate_dataset(x_val, y_val, num_segments_val, tokenizer)
    train_dataloader = DataLoader(
        train_dataset, batch_size=arg.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=arg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRU_BERT(arg.bert_model, arg.n_class, 0.3, arg.hid_gru_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arg.lr)
    CE_Loss = nn.CrossEntropyLoss()
    es = EarlyStopping(3, path=(arg.root_path + arg.name_model))
    for i in range(arg.epochs):
        loss = train_gru_fn(train_dataloader, model, optimizer, CE_Loss, device)
        output, target = eval_gru_fn(val_dataloader, model, device)
        accuracy = sum(np.array(output) == np.array(target)) / len(target)
        print(
            "epochs {} / {}  train_loss : {}  val_acc : {}".format(
                i + 1, arg.epochs, loss, accuracy
            )
        )
        es(accuracy, model, optimizer)

    test_df = pd.read_excel(arg.fig_root + "data/news.xlsx")
    x_test, y_test, num_segments_test = load_segments(
        test_df, vncore_tokenizer, arg.size_segment, arg.size_shift
    )
    test_dataset = generate_dataset(x_test, y_test, num_segments_test, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=arg.batch_size, shuffle=False)
    output_test, target_test = eval_fn(test_dataloader, model, device)
    test_acc = sum(np.array(output_test) == np.array(target_test)) / len(target_test)
    print("Accuracy test: ", test_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--size_segment", type=int, default=200)
    parser.add_argument("--size_shift", type=int, default=50)
    parser.add_argument("--bert_model", type=str, default="vinai/phobert-base")
    parser.add_argument("--bert_tokenizer", type=str, default="vinai/phobert-base")
    parser.add_argument(
        "--vncore_tokenizer", type=str, default="./vncorenlp/VnCoreNLP-1.1.1.jar"
    )
    parser.add_argument("--hid_gru_dim", type=int, default=200)
    parser.add_argument("--name_model", type=str, default="gru_bert")
    parser.add_argument("--test_size", type=int, default=0.2)
    parser.add_argument("--n_class", type=int, default=4)
    parser.add_argument("--root_path", type=str, default="./")
    args = parser.parse_args()
    main(args)
