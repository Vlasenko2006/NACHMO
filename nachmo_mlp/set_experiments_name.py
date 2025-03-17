import torch

def set_experiments_name(cfg):



    loss_config = cfg.loss_config
    stepper_config = cfg.stepper_config
    net_config = cfg.net_config
    data_config = cfg.data_config
    loader_config = cfg.loader_config
    train_config = cfg.train_config
    experiment_config = cfg.experiment_config
    visualization_config = cfg.visualization_config
    log_name = cfg.log_name
    Exp_name = cfg.exp_name
    extra_option = cfg.extra_option
    description = cfg.description
    depth = data_config.ntimesteps_in_training_set
    data_dtype = eval(data_config.data_type)
    NN_dtype = eval(net_config.NN_dtype)
    curriculum_scheme = cfg.curriculum_scheme + [data_config.trajectory_length]

    curr="Curr_"
    sub = "_"
    sur = "_"
    if len(curriculum_scheme) == 1: curr="No_curr_"
    if data_config.subtarct_mean==True: sub = "_sub"
    if stepper_config.try_Ssur_matrix==True: sur = "_Ssur"


    Exp = (
        Exp_name
        + curr
        + "_" + str(data_dtype)
        + sur
        + sub + "_mean_"
        + "_tr_length_"
        + str(data_config.trajectory_length)
        + "_opt_" + str(extra_option)
    )

    return Exp
