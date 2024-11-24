import json


class Config:
    def __init__(self, args, is_bert=False):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        with open(f'{config["bert_name"]}/config.json', "r", encoding="utf-8") as f:
            bert_config = json.loads(f.read())
        for k, v in bert_config.items():
            self.__dict__[k] = v
        self.epochs = config["epochs"]
        self.latent = config["latent"]
        self.promptLen = config["promptLen"]
        self.batch_size = config["batch_size"]
        self.gpu_id = config["gpu_id"]
        self.lstm_hid_size = config["lstm_hid_size"]
        self.height = config["height"]
        self.width = config["width"]

        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]

        self.bert_name = config["bert_name"]
        self.vit_name = config["vit_name"]
        self.bert_learning_rate = config["bert_learning_rate"]
        self.warm_factor = config["warm_factor"]
        self.max_sequence_length = config["max_sequence_length"]
        self.lang = config["lang"]
        self.bert_hid_size = config["bert_hid_size"]
        self.feature_hidden = config["feature_hidden"]
        self.dropout = config["dropout"]
        self.use_crf = config["use_crf"]
        self.cluster_threshold = config["cluster_threshold"]

        self.alpha = config["alpha"]
        self.logPath = config["logPath"]
        self.model_name = config["model_name"]

        self.seed = config["seed"]
        self.is_miss = config["is_miss"]
        self.is_h = config["is_h"]


        for k, v in args.__dict__.items():
            if v is not None:
                self.__dict__[k] = v

    def __repr__(self):
        return "{}".format(self.__dict__.items())