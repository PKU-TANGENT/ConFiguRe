import enum
from lib2to3.pgen2 import token
from typing import Dict
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import os
import re
import json
from cut_sent import cut_len, cut_sent

punc_list = ["。", "！", "？", "，", "；", ",", "?", "!", "——", ";", ":", "："]


def get_span_list(text: str):

    # res_list = list()
    # prev = 0
    # for idx, token in enumerate(text):
    #     if token in punc_list:
    #         res_list.append(text[prev:idx+1])
    #         prev = idx+1

    res_list = cut_sent(text)

    return res_list

# functions that overwrites ends with "_"


def overwrite_fig_label_(args, text_list, unit, labels, sent_pos):

    fig_name = unit["fos"]
    fig_start, fig_end = unit["begin"], unit["end"]
    tmp_start = 0
    while tmp_start < len(labels) - 1 and sent_pos[tmp_start] < fig_start:
        tmp_start += 1
    tmp_end = tmp_start + 1# tmp end marks the start of the next sentence

    while tmp_end < len(labels) and sent_pos[tmp_end] < fig_end:
        tmp_end += 1

    labels[tmp_start] = "B"
    labels[tmp_start+1:tmp_end] = ["I"] * (tmp_end-tmp_start-1)


def get_data(args, split):
    data_path = None
    if split == "train":
        data_path = args.train_data_path
    elif split == "valid":
        data_path = args.valid_data_path
    else:
        data_path = args.test_data_path

    with open(data_path, 'r', encoding="utf-8") as fin:
        data = json.load(fin)

    x_list = list()  # list of contexts, each element is a context (type of which is string)
    y_list = list()  # list of labels, each element is a list of labels
    sent_pos_list = list()

    for key in data:
        val = data[key]
        content = val["fragment"].strip()
        span_list = get_span_list(content)
        labels = ["O"] * len(span_list)
        sent_len = [len(x) for x in span_list]
        sent_pos = [0]
        cur_pos = 0
        for item in sent_len:
            sent_pos.append(cur_pos + item)
            cur_pos += item
        # overwrite labels according to fig units
        for unit in val["units"]:
            overwrite_fig_label_(args, span_list, unit, labels, sent_pos)

        x_list.append(span_list)
        y_list.append(labels)
        sent_pos_list.append(sent_pos)

    return x_list, y_list, sent_pos_list


class FigDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, sent_pos, sent_ori_len):
        self.encodings = encodings
        self.labels = labels
        self.sent_pos = sent_pos
        self.sent_ori_len = sent_ori_len
        self.sent_ids = []
        # self.get_punc_idx()
        self.get_sent_ids()
        assert len(self.encodings["input_ids"]) == len(self.labels)
        assert len(self.sent_ids) == len(self.labels)

    def get_sent_ids(self):
        for i, enc in enumerate(self.encodings['input_ids']):
            offset = self.encodings['offset_mapping'][i]
            reverse_offset = np.ones(self.sent_ori_len[i], dtype=np.int) * -1
            for j, span in enumerate(offset):
                reverse_offset[span[0]:span[1]] = j

            cur_pos = 1
            cur_sent_id = 1
            sent_ids = np.zeros((len(enc)))
            for j in range(len(self.sent_pos[i])-1):
                if self.sent_pos[i][j+1]-1 >= len(reverse_offset):
                    sent_ids[reverse_offset[self.sent_pos[i][j]]:] = cur_sent_id
                    break
                else:
                    sent_ids[reverse_offset[self.sent_pos[i][j]]                             :reverse_offset[self.sent_pos[i][j+1]-1]+1] = cur_sent_id
                    cur_sent_id += 1
            self.sent_ids.append(sent_ids)

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['sent_ids'] = torch.tensor(self.sent_ids[idx]).long()
        # item['punc_idx'] = torch.tensor(self.punc_idx[idx]).long()
        return item


def merge_data(key: str, data_list):
    """
    helper function to facilitate function `fig_collate_fn`
    parameter `data_list` is same as below

    """

    val_list = list()
    maxlen = 0

    key2pad = {"input_ids": 0, "token_type_ids": 0,
               "attention_mask": 0, "labels": -1, "sent_ids": 0}

    for data in data_list:
        maxlen = max(maxlen, data[key].shape[0])
        val_list.append(data[key].tolist())

    for idx, val in enumerate(val_list):
        val_list[idx].extend([key2pad[key]]*(maxlen - len(val)))

    return torch.tensor(val_list)


def fig_collate_fn(data_list):
    """
    Customized collate function to merge a list of data from FigDataset
    into a batch. Mainly performing in-batch padding.

    datalist: a list, each element is a dict:
        {
            "input_ids": [xxxxx]
            "token_type_ids": [xxxxx]
            "attention_mask": [xxxxx]
            "labels": [xxxxx]
            "punc_idx": [xxxxx]
        }
    """

    result = dict()
    for key in ["input_ids", "token_type_ids", "attention_mask", "labels", "sent_ids"]:
        result[key] = merge_data(key, data_list)

    return result


def load_data_split(args, tokenizer, split):

    data_x, data_y, sent_pos = get_data(args, split)
    data_x = ["".join(x) for x in data_x]

    # we leave padding to Dataloader, for now just get token embedding
    data_x_emb = tokenizer(data_x, padding=False,
                           truncation=True, return_offsets_mapping=True)
    data_y_num = [[args.lb2id[y] for y in y_list] for y_list in data_y]

    fig_dataset = FigDataset(data_x_emb, data_y_num, sent_pos, [len(x) for x in data_x])
    args.logger.info(f"{split} size: {len(fig_dataset)}")

    return fig_dataset

def load_dataset(args, tokenizer):

    train_dataset = load_data_split(args, tokenizer, "train")
    valid_dataset = load_data_split(args, tokenizer, "valid")
    test_dataset = load_data_split(args, tokenizer, "test")

    return train_dataset, valid_dataset, test_dataset
