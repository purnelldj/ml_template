# ml_template

template for ML projects

## initializing repository

search all references of `ml_template` and change it to the desired repo name.

## installation

Download and cd into repository and then:

```
python -m venv .venv
source venv/bin/activate
python -m pip install -e .
```

## usage

The main entry point is in [src/traintest.py](src/traintest.py), you can use it from the command line by typing

```
traintest
```
or to test

```
traintest stage=test ckpt_path=path/to/ckpt.ckpt
```
you could also change parameters from the command line, for example

```
traintest model.optimizer.lr=0.001
```
a hierarchy of configuration files is found in [src/conf](src/conf) and the default parameters are stored in [src/conf/main.yaml](src/conf/main.yaml).

## logging

to log to wandb first check that you are logged in

```
wandb login
```
then the config file and results will be uploaded to wandb for every run


## adding additional models / datasets

By default, data should be stored in a directory `ml_template/datasets`. If you would like to add a new dataset, you need to add a new data module that inherits from the `BaseDM` class and a new cofig file to instantiate the datamodule.
Similarly, to add a new model to work with [src/traintest.py](src/traintest.py), you need to add a new model class that inherits from the `BaseModel` class and a corresponding config (yaml) file to instantiate the model class.
