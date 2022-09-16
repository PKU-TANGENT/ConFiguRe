import torch
def task_setup(args, config):

    task_dict = {
        "dataset" : "DatasetForCRF",
        "model" : "BertForFigRecognitionCRF",
        "metric" : "MetricForRecognitionCRF",
        "train" : "TrainRecognitionCRF"
    }

    args.task_dict = task_dict

    # args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.figs = ["比喻","比拟","借代","夸张","反语","通感","问语","排比","对偶","反复","对比","引语"]
    args.id2lb = []
    if args.crf:
        args.id2lb += ["<start_token>", "<end_token>"] # default add start and end token to the start of labels
    args.id2lb += ["O"]
    for fig in args.figs:
        args.id2lb += [f"B-{fig}"]
        args.id2lb += [f"I-{fig}"]
    args.lb2id={}
    for i, lb in enumerate(args.id2lb):
        args.lb2id[lb] = i
    # args.lb2id = {"O":0}
    # for idx, fig in enumerate(args.figs):
    #     args.lb2id[f"I-{fig}"] = idx+1
    args.num_labels = len(args.id2lb)