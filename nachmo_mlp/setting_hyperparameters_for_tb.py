from chemical_constants_and_parameters import S_oh, S_verwer


def setting_hyperparameters_for_tb(cfg):

    cfg1 = {}
    cfg1["layers"] = {}
    cfg1["activations"] = {}
    for i in range(0, len(cfg.net_config.n_hidden)):
        cfg1["layers"][str("layer " + str(i + 1))] = cfg.net_config.n_hidden[i]
    cfg1["activations"][str(cfg.net_config.activation)] = True

    cfg["layers"] = cfg1["layers"]

    cfg["lr"] = cfg.train_config.lr

    if cfg.net_config.activation == "Sigmoid":
        cfg["activations"]["Sigmoid"] = 1
    if cfg.net_config.activation == "ReLU":
        cfg["activations"]["ReLU"] = 1
    if cfg.net_config.activation == "PReLU":
        cfg["activations"]["PReLU"] = 1
    return cfg

