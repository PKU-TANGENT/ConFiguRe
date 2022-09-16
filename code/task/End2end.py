def task_setup(args, config):

    task_dict = {
        "dataset" : "DatasetForRecognition",
        "model" : "BertForFigRecognition",
        "metric" : "MetricForRecognition",
        "train" : "TrainRecognition"
    }

    args.task_dict = task_dict

    args.figs = ["比喻","比拟","借代","夸张","反语","通感","问语","排比","对偶","反复","对比","引语"]
    args.id2lb = ["O"]
    for fig in args.figs:
        args.id2lb += [f"B-{fig}"]
        args.id2lb += [f"I-{fig}"]
    args.lb2id = {"O":0}
    for idx, fig in enumerate(args.id2lb):
        args.lb2id[fig] = idx
    args.num_labels = len(args.id2lb)
    config.num_labels = args.num_labels