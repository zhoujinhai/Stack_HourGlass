import argparse


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', type=str, default='tooth', help='experiments name')
    parser.add_argument('-m', '--max_iters', type=int, default=200, help='max number of iterations ')
    parser.add_argument("-b", '--batch', type=int, default=4, help='the batch size of train data')
    parser.add_argument('-ptm', '--pre_trained_model', type=str, help='the preview trained model')
    parser.add_argument('-s', '--save_frequence', type=int, default=10, help='the frequence of save the trained model')
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.001, help="the learning rate in training")
    parser.add_argument('-r', '--train_data_ratio', type=float, default=0.9, help="the training data ratio of all datas")
    args = parser.parse_args()
    return args


__config__ = {
    # -----NetWork-----
    "n_stack": 3,
    "in_dim": 256,
    "n_kp": 3,
    "n_hg_layer": 3,

    # -----Train-----
    "device": "cpu",  # "cuda"
    "epoch_num": 200,
    "start_epoch": 0,
    "lr": 0.001,
    "batch_size": 4,
    "pre_trained_model": "./exp/tooth/checkpoint.pt",
    "save_freq": 10,

    # -----Data-----
    "img_dir": r"E:\data\SplitTooth\AddFDIClassAndKeyPoint\keyPoint\black",
    "label_file": r"E:\data\SplitTooth\AddFDIClassAndKeyPoint\keyPoint\kp_label.txt",
    "train_ratio": 0.9,                       # ratio of train data in all data

    # -----Exp-----
    "exp": "tooth"
}


def get_config():
    opt = parse_command_line()

    config = __config__
    config["epoch_num"] = opt.max_iters
    config["batch_size"] = opt.batch
    if opt.pre_trained_model is not None:
        config["pre_trained_model"] = opt.pre_trained_model
    config["save_freq"] = opt.save_frequence
    config["lr"] = opt.learning_rate
    config["train_ratio"] = opt.train_data_ratio
    config["exp"] = opt.exp

    return config


if __name__ == "__main__":
    conf = get_config()
    print(conf)
