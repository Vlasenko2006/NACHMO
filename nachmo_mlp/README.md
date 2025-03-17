# NACHMO code

Here is a brief summary on the code content and instructions on how to install it and run the experiments


## Code content
- chemical_constants_and_parameters.py &nbsp;  &nbsp; &nbsp; _Contains stoichiometry matrices_
- dataloader.py.    &nbsp;  &nbsp; &nbsp; _arranges the dataset in batches and distributes it among the devices (cpus, gpus)_.
- dataset.py        &nbsp;  &nbsp; &nbsp;     _Prepares a dataset consisting of input (chemical states) and outputs (chemical state sequences)_
- exec.bash                    &nbsp;  &nbsp; &nbsp;   _Contains slurm commands for multiprocessor run_
- loss_function.py             &nbsp;  &nbsp; &nbsp;   _contains loss function._
- main.py                       &nbsp;  &nbsp; &nbsp;  _Th driver program_.
- metrics.py             &nbsp;  &nbsp; &nbsp;   _Computes MAE and MRE for validation data_ .
- models.py                     &nbsp;  &nbsp; &nbsp;  _Creates the MLP and the stepper._
- model_calculation.py         &nbsp;  &nbsp; &nbsp;   _Evaluates model several timesteps ahead, preserving autograd tape (code dependencies)._
- train.py                     &nbsp;  &nbsp; &nbsp;  _Contains training and logger subroutines._ 
- utilities.py                 &nbsp;  &nbsp; &nbsp;   _Loads the data from the harddrive, normalizes it. Returns tensors containing concentrations and normalization factors. This data is used by_ `dataset.py`
- visualization.py &nbsp;  &nbsp; &nbsp; _ Visualizes NN outputs on the fly during the training _ 

## 2. Setting `config.yaml` file

open `config.yaml` In this file find the lists given below and change the default values, if needed. Each item cooresonds to a NN hyperparamter or to the training settings. The names are self-explainable. Anyhow, see comments starting with `#` opposite to corresponding item for more info.  

- **hardw_settings**:
     - devices: 1.   &nbsp;  &nbsp; &nbsp; # _num of GPU's per node_ 
     - accelerator: "gpu" &nbsp;  &nbsp; &nbsp; # _change to "cpu" if run on cpu nodes_
     - strategy: "ddp"
     - num_of_nodes: 1

- **train_config**:
     - lr: 1e-6 &nbsp;  &nbsp; &nbsp; # _learning rate_
     - n_epochs: 1240 &nbsp;  &nbsp; &nbsp; # _number of epochss_

- **loader_config**:
     - batch_size: 256
     - num_workers: 1 &nbsp;  &nbsp; &nbsp; # _don't touch_ 
 
- **experiment_config**:
     - val_frac: 0.11 &nbsp;  &nbsp; &nbsp; # _fraction of validation data_
     - test_frac: 0.09  &nbsp;  &nbsp; &nbsp; # _fraction of test data_

- **data_config**:
     - scheme: "Verwer" &nbsp;  &nbsp; &nbsp; # _this flag set to compute Verwer scheme. For dynamic OH chane it to "OH"_
     - species: 
     - species_oh: ["OH", "HO2", "H2O2"]  &nbsp;  &nbsp; &nbsp; #  _don't touch !_
     - species_verwer: ["CO","HNO3","SO4","XO2",... &nbsp;  &nbsp; &nbsp; #  don't touch !
     - trajectory_length: 2  &nbsp;  &nbsp; &nbsp; # _length of multistep training_
- **net_config**:
     - n_hidden: [40,40,40] &nbsp;  &nbsp; &nbsp; # _Number of elements in this list corresponds to NN depth, the value of the element is the amount of neurons in the corresponding level_
    - input_products: True  &nbsp;  &nbsp; &nbsp; # _flag defining whether we provide all concentration products as input_
    - activation: "ReLU"
    - device: "cuda"  &nbsp;  &nbsp; &nbsp; # _If you compute on CPU node, change to "cpu"_ 
    - depth: 900 #59999  &nbsp;  &nbsp; &nbsp; # _Corresponds to the time length (in time steps) of the training data. Set to the maximal number of timesteps if you whant ot train on the whole set_
    - debug: False

- **stepper_config**:
    - learn_updates: True. &nbsp;  &nbsp; &nbsp; # 
    - learn_rates: False
    - stoichiometry_matrix: False &nbsp;  &nbsp; &nbsp; # If set to the `True`, set `learn_rates: True` and `learn_updates: False`.

- **exp_name**: "Exp" &nbsp;  &nbsp; &nbsp; # _Name of the experiment_
- **log_name**: "nul" &nbsp;  &nbsp; &nbsp; # _Name of the directory where the NN stores checkpoints, model, results and hyperparameters._
- **path_to_data**:

