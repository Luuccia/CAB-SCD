# CAB-SCD

The pytorch implementation for CAB-SCD: Cost Aggregation-Based model for Semantic Change Detection.

## Requirements
All experiments are done on NVIDIA GeForce RTX 3090.

* Python 3.10.16
* Pytorch 1.11.0

## Install

1. Create Conda environment

    ```
    conda create -n ovcd python=3.10
    ```

2. Install PyTorch

    ```
    # install pytorch 1.11.0
    conda install pytorch=1.11.0 torchvision=0.12.0 torchaudio=0.11.0 cudatoolkit=11.3 -c pytorch
    ```

3. Install other dependencies

    ```
    pip install -r requirements.txt

    cd open_clip
    make install
    ```

## Datasets Preparation

Refer to the structure of each dataset under the `data` folder and use the python scripts with a "deal" prefix to perform data processing.

# Train

```
python train.py --config config/path
```

# Test

```
python eval.py --checkpoint weight/path -config config/path
```