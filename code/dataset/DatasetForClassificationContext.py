from functools import reduce
import torch
import json

def reduce_helper_func(x_ctxt_y_list, val):
    fig_list = val["units"]
    fragment = val["fragment"]
    def map_helper_func(fig):
        st = fig["begin"]
        ed = fig["end"]
        return (fig["figurativeUnit"], fragment[max(0, st-50):min(len(fig["figurativeUnit"]),ed+50)], fig["fos"])
    to_extend_x, to_extend_ctxt, to_extend_y = zip(*map(map_helper_func, fig_list))
    x_ctxt_y_list[0].extend(to_extend_x)
    x_ctxt_y_list[1].extend(to_extend_ctxt)
    x_ctxt_y_list[2].extend(to_extend_y)
    return x_ctxt_y_list


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
    data_list = [data[i] for i in data]
    x_ctxt_y_list = [[], [], []]

    x_ctxt_y_list = reduce(reduce_helper_func, data_list, x_ctxt_y_list)

    return (x_ctxt_y_list[0], x_ctxt_y_list[1]), x_ctxt_y_list[2]


class FigDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings[0]
        self.context = encodings[1]
        self.labels = labels
        # self.get_punc_idx()
        assert len(self.encodings["input_ids"]) == len(self.labels)

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        for key, val in self.context.items():
            item[f"{key}2"] = torch.tensor(val[idx])
        # item['punc_idx'] = torch.tensor(self.punc_idx[idx]).long()
        return item


def merge_data(key: str, data_list):
    """
    helper function to facilitate function `fig_collate_fn`
    parameter `data_list` is same as below

    """

    val_list = list()
    maxlen = 0

    key2pad = {"input_ids": 0, "token_type_ids": 0, "attention_mask": 0,
               "labels": -1, "input_ids2": 0, "token_type_ids2": 0, "attention_mask2": 0}

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
        }
    """

    result = dict()
    for key in ["input_ids", "token_type_ids", "attention_mask", "labels", "input_ids2", "token_type_ids2", "attention_mask2"]:
        result[key] = merge_data(key, data_list)

    return result


def load_data_split(args, tokenizer, split):

    data_x, data_y = get_data(args, split)

    # we leave padding to Dataloader, for now just get token embedding
    data_x_emb0 = tokenizer(data_x[0], padding=False,
                            truncation=True, return_offsets_mapping=True)

    data_x_emb1 = tokenizer(data_x[1], padding=False,
                            truncation=True, return_offsets_mapping=True)

    data_y_num = [[args.lb2id[y]] for y in data_y]

    fig_dataset = FigDataset((data_x_emb0, data_x_emb1), data_y_num)
    args.logger.info(f"{split} size: {len(fig_dataset)}")

    return fig_dataset


def load_dataset(args, tokenizer):

    train_dataset = load_data_split(args, tokenizer, "train")
    valid_dataset = load_data_split(args, tokenizer, "valid")
    test_dataset = load_data_split(args, tokenizer, "test")

    return train_dataset, valid_dataset, test_dataset
