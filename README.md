# Quantum device tuning via hypersurface sampling

The quantum devices used to implement spin qubits in semiconductors can be challenging to tune and characterise. Often the best approaches to tuning such devices is manual tuning or a simple heuristic algorithm which is not flexible across devices. This repository contains the statistical tuning approach detailed in https://arxiv.org/abs/2001.02589 with some additional functionality. This approach is promising as it make few assumptions about the device being tuned and hence can be applied to many systems without alteration.

## Dependencies
The required packages required to run the algorithm are:
```
scikit_image
scipy
numpy
matplotlib
GPy
mkl
pyDOE
skimage
```
# Using the algorithm
Using the algorithm varies depending on what measurement software you use in your lab or what you want to achieve. Specifically if your lab utilises pygor then you should call a different function to initiate the tuning. If you are unable to access a lab then you can still create a virtual environment to test the algorithm in using the Playground module. Below is documentation detailing how to run the algorithm for each of these situations.
## Without pygor
To use the algorithm without pygor you must create the following:
- jump
- measure
- check
- config_file

<ins>jump:</ins>
Jump should be a function that takes an array of values and sets them to the device. It should also accept a flag that details whether the investigation gates (typically plunger gates) should be used. Below is an example of how jump should be defined for a 5 gate device with 2 investigation (in this case plunger) gates.
```python
def jump(params,inv=False):
  if inv:
    labels = ["dac4","dac6"] #plunger gates
  else:
    labels = ["dac3","dac4","dac5","dac6","dac7"] #all gates
    
  assert len(params) == len(labels) #params needs to be the same length as labels
  for i in range(len(params)):
    set_value_to_dac(labels[i],params[i]) #function that takes dac key and value and sets dac to that value
  return params
```
<ins>measure:</ins>
measure should be a function that returns the measured current on the device.
```python
def measure():
  current = get_value_from_daq() #receive a single current measurement from the daq
  return current
```
<ins>check:</ins>
check should be a function that returns the state of all relevant dac channels.
```python
def check():
  labels = ["dac3","dac4","dac5","dac6","dac7"] #all gates
  dac_state = [None]*len(labels)
  for i in range(len(labels)):
    dac_state[i] = get_current_dac_state(labels[i]) #function that takes dac key and returns state that channel is in
  return dac_state
```
<ins>config_file:</ins>
config_file should be a string that specifies the file path of a .json file containing a json object that specifies the desired settings the user wants to use for tuning. An example string would be "demo_config.json". For information on what the config file should contain see the json config section.

### How to run
To run tuning without pygor once the above has been defined call the following:
```python
import AutoDot
AutoDot.tune.tune_from_file(jump,measure,check,config_file)
```
## With pygor
To use the algorithm without pygor you must create the following:

<ins>config_file:</ins>
config_file should be a string that specifies the file path of a .json file containing a json object that specifies the desired settings the user wants to use for tuning. An example string would be "demo_config.json". For information on what the config file should contain see the json config section. Additional fields are required to specify pygor location and setup.
### How to run
To run tuning with pygor once the above has been defined call the following:
```python
import AutoDot
AutoDot.tune.tune_with_pygor_from_file(config_file)
```
## With playground (environment)
To use the algorithm using the playground you must create the following:

<ins>config_file:</ins>
config_file should be a string that specifies the file path of a .json file containing a json object that specifies the desired settings the user wants to use for tuning. An example string would be "demo_config.json". 

The config you must supply the field "playground" then in this field you must specify the basic shapes you want to build your environment out of. Provided is a [demo config file](mock_device_demo_config.json) and a [README](Playground/README.md) detailing how it works and what a typical run looks like.

### How to run
To run tuning with pygor once the above has been defined call the following:
```python
import AutoDot
AutoDot.tune.tune_with_playground_from_file(config_file)
```

# Config structure
Here is an [example config file](demo_config.json) containing all the relevant fields and below is a discussion about each fields function
```
"path_to_pygor":"/path_to/pygor/package"
```
Absolute path to pygor package. (only required if using pygor)
```
"ip":"http://123.123.123.12:8000/RPC2"
```
URL of pygor server. If not specified mock device is built (only required if using pygor)
```
"gates":["c10","c3","c4","c5","c6","c7","c9"]
```
Labels for the DAC channels that the gates are attached to. (only required if using pygor)
```
"plunger_gates":["c5","c9"]
```
Labels for the DAC channels that the investigation/plunger gates are attached to. (only required if using pygor)
```
"bias_chans":["c1"]
```
Label(s) for the DAC channel(s) that control the bias. (only required if using pygor)
```
"bias_vals":[20]
```
Value(s) to set the bias DAC channel(s) to. (only required if using pygor)
```
"chan_no":0
```
0 based index of the channel that is used to measure the device. (only required if using pygor)
```
"save_dir":"save_file_name"
```
Relative path to save tuning data in.
