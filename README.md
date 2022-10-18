# LCI-Net

Implementation of the IJRR paper: 

Locally Connected Interrelated Network: A Forward Propagation Primitive

The code implements the LCI-Net module embedded in VIN and QMDP-net architectures.


### Requirements

Python 3.6 or newer (tested up to Python 3.9)
Tensorflow 1.14 or newer (tested up to Tensorflow 2.7)
Python packages: numpy, scipy, pillow

To install these packages using pip:
```
pip install tensorflow
pip install numpy scipy pillow
```

### Train and evaluate networks

```
python train.py [input_file] --network_type [network_type] --logpath [dir_to_save_models]
```
The given input_file must be a file produced by env_generator.py (or an empty file of the same format, e.g. for
imagegrid large scale realistic envs).

If the input file env type is fully observable (i.e. name contains 'full_obs'), then network type must be 'vin_baseline'
or 'vin_lcinet'. Otherwise, network type must be 'qmdpnet_baseline' or 'qmdpnet_lcinet').

e.g.
```
python train.py inputs/env_grid_nondet_10x10_example.txt --network_type qmdpnet_lcinet --logpath outputs/example
```

To continue training an already existing model, a loadmodel path can be specified:
```
python train.py [input_file] --network_type [network_type] --loadmodel [dir_of_saved_model] --logpath [dir_to_save_models]
```
The loadmodel path should be the same as the logpath used when the existing model was first trained, e.g. 
```
python train.py inputs/env_grid_nondet_10x10_example.txt --network_type qmdpnet_lcinet --loadmodel outputs/example --logpath outputs/continued
```

An existing model can be evaluated individually by setting epochs to zero, e.g.
```
python train.py [input_file] --network_type [network_type] --loadmodel [dir_of_saved_model] --epochs 0
```

### Compare LCI-Net networks with original networks

If networks have been trained for both network types (LCI-net and original), a comparison between both types on the same
set of maps and trajectories can be performed via:
```
python evaluate.py [input_file] --loadmodel [dir_of_saved_models]
```

The number of maps, trajectories and repetitions can be manually specified - refer to arguments.py to see available
arguments.


### Generate Training Data

Environment files can be generated via the following command:
```
python env_generator.py [env_type] [num_maps] [trajs_per_map] [name_suffix]
```

e.g. to generate an input file with 2000 dynamic maze v2 maps with 5 trajectories per map with suffix 'example':
```
python env_generator.py dynmaze_v2_nondet 2000 5 example
```

Refer to env_generator.py to find the supported env_type values.

