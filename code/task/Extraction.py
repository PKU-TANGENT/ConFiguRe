def task_setup(args, config):

    task_dict = {
        "dataset" : "DatasetForExtraction",
        "model" : "BertForFigExtraction",
        "metric" : "MetricForExtraction",
        "train" : "TrainExtraction"
    }

    args.task_dict = task_dict

    args.lb2id = {"O": 0, "B": 1, "I": 2}
    args.id2lb = ["O", "B", "I"]
    args.num_labels = len(args.id2lb)

    config.num_labels = args.num_labels