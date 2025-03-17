def net_info(train_data,valid_data,test_data, cfg, Exp, path):


    print(
        "train_data.shape = ",
        train_data.shape,
        "valid_data.shape = ",
        valid_data.shape,
        "test_data.shape = ",
        test_data.shape,
    )

    print("scheme = ", cfg.data_config.scheme, "train_set_timesteps = ", cfg.data_config.ntimesteps_in_training_set, "data_path = ", cfg.data_config.data_path)
    print("trajectory_lenth = ", cfg.data_config.trajectory_length)
    print("Net_config", cfg.net_config)
    print("Experiment_name = ", Exp)
    print("path to the output = ", path)
    print("description", cfg.description)

