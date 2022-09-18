import os
import argparse
import logging
import torch
import random
import importlib
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertConfig
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig
from types import SimpleNamespace

def setup_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    # StreamHandler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    return logger

def sanity_check(args, data_x, data_y, data_x_emb, tokenizer):
    """
    sanity check to avoid fundamental mistakes
    """
    print(data_x[0])
    print(data_y[0])
    print(data_x_emb["input_ids"][0])

    sep_id = tokenizer.sep_token_id

    total_num = len(data_x)
    error_num = 0

    for i in range(total_num):
        if data_x_emb["input_ids"][i].count(sep_id) != len(data_y[i]):
            error_num += 1
            print(f"Inconsistent in data {i}")
            if error_num < 5:
                print(data_x[i])
                print(data_x_emb["input_ids"][i])
                print(data_y[i])

    args.logger.info(f"Progress| Sanity check passed! Error num is {error_num}")

@hydra.main(config_path="configs", config_name="default")
def main(cfg:DictConfig):
    # Set random seed

    args=SimpleNamespace(**cfg.model_args)
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Get accelerator
    task_setup = getattr(importlib.import_module(f"..{args.task}", package="task.subpkg"), "task_setup")
    accelerator = Accelerator()
    args.device = accelerator.device

    args.logger = setup_logger(args)
    args.logger.info("Progress| Parse args succeed!")

    # Set config
    config = BertConfig.from_pretrained(args.model_name_or_path)
    args.logger.info("Progress| Load Bert config succeed!")

    # Task setup
    task_setup(args, config)
    args.logger.info("Progress| Task setup succeed!")

    tokenizer = BertTokenizerFast.from_pretrained(
        args.model_name_or_path, add_special_tokens=False, model_max_length=512)
    args.logger.info("Progress| Load Bert tokenizer succeed!")

    load_dataset = getattr(importlib.import_module(f"..{args.task_dict['dataset']}", package="dataset.subpkg"), "load_dataset")
    train_dataset, valid_dataset, test_dataset = load_dataset(args, tokenizer)

    fig_collate_fn = getattr(importlib.import_module(f"..{args.task_dict['dataset']}", package="dataset.subpkg"), "fig_collate_fn")
    train_dataloader = DataLoader(
        train_dataset, collate_fn=fig_collate_fn, batch_size=args.batch_size)
    
    valid_dataloader = DataLoader(
        valid_dataset, collate_fn=fig_collate_fn, batch_size=args.eval_batch_size)  
    
    test_dataloader = DataLoader(
        test_dataset, collate_fn=fig_collate_fn, batch_size=args.eval_batch_size)  

    # Get model
    figmodel = getattr(importlib.import_module(f"..{args.task_dict['model']}", package="model.subpkg"), f"{args.task_dict['model']}")
    model_init_dict = {}
    for key in args.model_init_args:
        model_init_dict[key] = eval(key)
    model = figmodel.from_pretrained(args.model_name_or_path, **model_init_dict).to(args.device)
    args.logger.info("Progress| Load Bert model succeed!")

    # Get optimizer
    if args.optimizer == "default":
        optimizer = AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer_list = []
        for parameters in args.optimizer:
            optimizer_list.append({"params": eval(parameters["params"]), "lr": parameters["lr"]})
        optimizer = AdamW(optimizer_list)

    args.logger.info("Progress| Load Optimizer AdamW succeed!")

    train_dataloader, valid_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, valid_dataloader, model, optimizer
    )
    args.logger.info("Progress| Load Accelerator succeed!")

    # Train
    train_fun = getattr(importlib.import_module(f"..{args.task_dict['train']}" , package="train.subpkg"), f"{args.task_dict['train']}")
    Metric = getattr(importlib.import_module(f"..{args.task_dict['metric']}" , package="metrics.subpkg"), f"{args.task_dict['metric']}")
    if not args.eval_only:
        train_fun(args, model, optimizer, train_dataloader, valid_dataloader, accelerator, Metric)

    # Evaluate on test set
    args.logger.info("Progress| Train Complete!")
    evaluation = getattr(importlib.import_module(f"..{args.task_dict['train']}" , package="train.subpkg"), "evaluation")
    evaluation(args, model, test_dataloader, -1, Metric,  os.path.join(args.save_dir, f"{args.task}_{args.arch_name}_ckp_best.pt"))


if __name__ == "__main__":
    main()
