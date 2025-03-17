from chemical_constants_and_parameters import S_oh, S_verwer


def chemical_mechanism_selector(cfg):


    if cfg.data_config.scheme == "Verwer":
        cfg.data_config.species = cfg.data_config.species_verwer
        Smatrix = S_verwer
        nrates=20

    if cfg.data_config.scheme == "OH":
        cfg.data_config.species = cfg.data_config.species_oh
        Smatrix = S_oh
        nrates = 4
    return Smatrix, nrates, cfg
