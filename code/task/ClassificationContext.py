def task_setup(args, config):

    task_dict = {
        "dataset" : "DatasetForClassificationContext",
        "model" : "BertForFigClassificationContext",
        "metric" : "MetricForClassification",
        "train" : "TrainClassifier"
    }

    args.task_dict = task_dict

    args.figs = ["比喻","比拟","借代","夸张","反语","通感","问语","排比","对偶","反复","对比","引语"]
    # random.shuffle(args.figs)
    args.id2lb = list()
    for fig in args.figs:
        args.id2lb += [fig]
    args.lb2id = dict()
    for idx, fig in enumerate(args.figs):
        args.lb2id[fig] = idx
    args.num_labels = len(args.id2lb)
    args.eval_batch_size = args.batch_size
    config.num_labels = args.num_labels
    

