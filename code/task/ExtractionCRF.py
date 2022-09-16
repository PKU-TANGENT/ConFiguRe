def task_setup(args, config):

    task_dict = {
        "dataset" : "DatasetForExtraction",
        "model" : "BertForFigRecognitionCRF",
        "metric" : "MetricForExtraction",
        "train" : "TrainRecognitionCRF"
    }

    args.task_dict = task_dict

    args.id2lb = ["<start_token>", "<end_token>", "O", "B", "I"]
    args.lb2id = dict()
    for i, lb in enumerate(args.id2lb):
        args.lb2id[lb] = i
    
    args.num_labels = len(args.id2lb)

    config.num_labels = args.num_labels