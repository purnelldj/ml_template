# ml_template

template for ML projects

## initializing repository

search all references of `ml_template` and change it to the desired repo name.

local data should be stored in `src/datasets/datasetname`.

run `pre-commit install`.

## installation

Download and cd into repository and then:

```
python -m venv .venv
source venv/bin/activate
python -m pip install -e .
```

## train and test

the default parameters are found in `src/conf/main.yaml`

```
traintest
```
or to test

```
traintest mode=test
```

you could also change model parameters from the command line, for example

```
traintest model.params.max_depth=20
```

## logging

to log to wandb first check that you are logged in

```
wandb login
```
then when running from the command line:

```
traintest log_to_wandb=True
```

then the config file and results will be uploaded to wandb for every run


## adding additional models / datasets

to add a new model to work with `traintest.py` , you need to add a new model class that inherits from the BaseModel class and a corresponding config (yaml) file to instantiate the model class.
If you would like to add a new dataset, you need to add a new data set and data module that inherits from pytorch lightning datamodule class.
