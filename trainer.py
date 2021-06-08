from tqdm.auto import tqdm


def train_fn(data_loader, model, optimizer, loss_fn, device):
    model.train()
    train_loss = 0
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        content_input_ids = d["content_input_ids"].to(device)
        content_token_type_ids = d["content_token_type_ids"].to(device)
        content_attention_mask = d["content_attention_mask"].to(device)
        label = d["label"].to(device)

        # content_input_ids = content_input_ids.to(device)
        # content_token_type_ids = content_token_type_ids.to(device)
        # content_attention_mask = content_attention_mask.to(device)
        # label = label.to(device)
        optimizer.zero_grad()
        outputs = model(
            content_input_ids=content_input_ids,
            content_attention_mask=content_attention_mask,
            content_token_type_ids=content_token_type_ids,
        )
        # print(outputs.shape)
        loss = loss_fn(outputs, label.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(data_loader)

