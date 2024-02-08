# crop classifier template
mila take-home exercise

## installation

download and cd into repository and then:

```
python -m venv .venv
source venv/bin/activate
python -m pip install -e .
```
## initializing repository

local data should be stored in `cropclassifier/datasets/datasetname` the trainval and test data files are contained in the same directory. For example `cropclassifier/datasets/snt12_era5_srtm/*.nc`

## train and test

the default parameters are found in `cropclassifier/cropclassifier/conf/main.yaml`

```
python cropclassifier/train_test.py
```
or to test

```
python cropclassifier/train_test.py mode=test
```

you could also change model parameters from the command line, for example

```
python cropclassifier/train_test.py model.params.max_depth=20
```

## logging

to log to wandb

```
python cropclassifier/train_test.py log_to_wandb=True
```

then the config file and results will be uploaded to wandb for every run


## adding additional models / datasets

to add a new model to work with train_test.py , you need to add a new model class that inherits from the BaseModel class and a corresponding config (yaml) file to instantiate the model class.
Similarly, if you would like to add a new dataset, you need to add a new data module class that inherits from the BaseDataMod class and add a corresponding config (yaml) file.
