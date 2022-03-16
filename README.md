# Sketches
Detect attributes of the sketches dataset. This dataset consists of drawings by designers and our goal is to find attributes of these drawings.

## Directory structure
Structure of the project
```
configs/
    params.yml
dist/
    Sketches-0.0.1-py3-none-any.whl
    Sketches-0.0.1.tar.gz
experiments/
    base_model/
        runs/
            train/
                events.out.tfevents
        best.pth.tar
        last.pth.tar
        params.yml
        train.log
notebooks/
    main.ipynb
scripts/
    upload_configs.sh
    upload_notebooks.sh
    uploadwhl.sh
src/
    model/
        __init__.py
        data_loader.py
        net.py
    utils/
        __init__.py
        dist_utils.py
        download_utils.py
        utils.py
    create_labels.py
    download_data.py
    evaluate.py
    train.py
tests/
    __init__.py
.gitignore
environments.yml
Makefile
pyproject.toml
README.md
setup.cfg
```

## Usage
First clone the project as follows,
```
git clone <url> <newprojname>
cd <newprojname>
```
Then build the project by using the following command, (assuming build is already installed in your virtual environment, if not then activate your virtual environment and use `conda install build`)
```
make build
```
Next, install the build wheel file as follows,
```
pip install <path to wheel file>
```
Then create the dataset by running the following command (this needs to be done only once, and can be done at anytime after cloning this repo),
```
python download_data.py -r <path to root dir>
python create_labels.py -r <path to root dir>
```
Then to start training on a single node with multiple gpu's we can do the following,
```
python <path to train.py> --args1 --args2
```

## Requirements
I used Anaconda with python3, but used pip to install the libraries so that they worked with my multi GPU compute environment in Azure

```
make install
conda activate sketches-env
```