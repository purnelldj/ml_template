# ml_template

template for ML projects

## installation

download and cd into repository and then:

```
python -m venv .venv
source venv/bin/activate
python -m pip install -e .
```
## initializing repository

search all references of `ml_template` and change it to the desired name.

local data should be stored in `ml_template/datasets/datasetname` the train / val and test data files are contained in the same directory.

## train and test

the default parameters are found in `ml_template/ml_template/conf/main.yaml`

```
python ml_template/train_test.py
```
or to test

```
python ml_template/train_test.py mode=test
```

you could also change model parameters from the command line, for example

```
python ml_template/train_test.py model.params.max_depth=20
```

## logging

to log to wandb first check that you are logged in

```
wandb login
```
then when running from the command line:

```
python ml_template/train_test.py log_to_wandb=True
```

then the config file and results will be uploaded to wandb for every run


## adding additional models / datasets

to add a new model to work with `train_test.py` , you need to add a new model class that inherits from the BaseModel class and a corresponding config (yaml) file to instantiate the model class.
Similarly, if you would like to add a new dataset, you need to add a new data module class that inherits from the BaseDataMod class and add a corresponding config (yaml) file.
