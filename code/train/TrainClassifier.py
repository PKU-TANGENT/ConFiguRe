import os
import torch
from transformers import get_scheduler
from tqdm.auto import tqdm


def TrainClassifier(args, model, optimizer, train_dataloader, eval_dataloader, accelerator, Metric):

    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    progress_bar = tqdm(range(num_training_steps))

    args.logger.info(f"Initial Evaluation Result")
    evaluation(args, model, eval_dataloader=eval_dataloader,
               epoch_num=-3, Metric=Metric)
    best_f1 = -1
    for epoch in range(num_epochs):
        model.train()
        for idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            if idx % 50 == 0:
                args.logger.info(
                    f"epoch {epoch}, iter {idx}, loss = {loss.data}, lr={lr_scheduler.get_last_lr()[0]}")

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        args.logger.info(f"Evaluation Result on epoch {epoch}:")
        f1_score = evaluation(
            args, model, eval_dataloader=eval_dataloader, epoch_num=epoch, Metric=Metric)

        if f1_score - best_f1 > 1e-10:
            best_f1 = f1_score
            args.logger.info(f"Saving model for checkpoint {epoch}")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        },  os.path.join(args.save_dir, f"{args.task}_{args.arch_name}_ckp_best.pt"))

    return


def evaluation(args, model, eval_dataloader, epoch_num, Metric, ckp_path=None):

    # if model is a path (i.e. to load a checkpoint)
    if ckp_path is not None:
        checkpoint = torch.load(ckp_path)
        args.logger.info(f"Load best checkpoint from epoch {checkpoint['epoch']}")
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    correct = 0
    total = 0
    metric = Metric(args.id2lb, args.lb2id, args.logger)
    for batch in eval_dataloader:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)  # bsz * seq_len
        preds = preds.view(-1, 1).squeeze(dim=-1)
        labels = batch["labels"].view(-1, 1).squeeze()
        correct += (preds == labels).sum().item()
        metric.add(preds.cpu().numpy().tolist(), labels.cpu().numpy().tolist())
        total += preds.shape[0]

    f1_score = metric.get_metrics(show_overall=True)

    # if epoch_num % 5 == 0:
    #     metric.save_confusion_matrix(epoch_num)

    return f1_score
