import torch
from tqdm import tqdm


def train_fn(data_loader, model, optimizer, loss_fn, device):
    model.train()
    train_loss = 0
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        content_input_ids = d["content_input_ids"].to(device)
        content_token_type_ids = d["content_token_type_ids"].to(device)
        content_attention_mask = d["content_attention_mask"].to(device)
        label = d["label"].to(device)
        optimizer.zero_grad()
        outputs = model(
            content_input_ids=content_input_ids,
            content_attention_mask=content_attention_mask,
            content_token_type_ids=content_token_type_ids,
        )
        loss = loss_fn(outputs, label.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["content_input_ids"]
            token_type_ids = d["content_token_type_ids"]
            mask = d["content_attention_mask"]
            targets = d["label"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)
            outputs = model(
                content_input_ids=ids,
                content_attention_mask=mask,
                content_token_type_ids=token_type_ids,
            )
            output = torch.argmax(outputs, dim=-1)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(output.cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets
