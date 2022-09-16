def task_setup(args, config):

    task_dict = {
        "dataset" : "DatasetForExtraction",
        "model" : "BertForFigExtractionContrast",
        "metric" : "MetricForExtraction",
        "train" : "TrainExtraction"
    }

    args.task_dict = task_dict

    args.lb2id = {"O": 0, "I": 1}
    args.id2lb = ["O", "I"]
    args.num_labels = len(args.id2lb)

    config.num_labels = args.num_labels
    config.contrast_lambda = args.contrast_lambda