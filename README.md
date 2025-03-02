

## <a name="0"></a> 0. Package Versions
* Packages
    ```
    conda install -c conda-forge tqdm
    conda install -c anaconda scipy
    conda install -c conda-forge scikit-learn
    conda install -c open3d-admin open3d
    conda install pytorch==1.4.0 torchvision==0.5.0 -c pytorch
    conda install -c huggingface transformers
    ```
* Follow instructions from https://github.com/facebookresearch/SparseConvNet to download SparseConvNet.

## <a name="1"></a> 1. Dataset Download

### Scannet Download
For the Scannet Dataset please go to https://github.com/ScanNet/ScanNet and fill out the agreement form to download the dataset.

### ScanRefer Download
For the ScanRefer Dataset please go to https://github.com/daveredrum/ScanRefer and fill out the agreement form to download the dataset.

### Glove Embeddings
Download the [*preprocessed glove embeddings*](http://kaldir.vc.in.tum.de/glove.p) from [*ScanRefer*](https://github.com/daveredrum/ScanRefer).

## <a name="2"></a> 2. Data Organization
```
scannet_data
|--scans

This Repository
|--glove.p
|--ScanRefer
    |--Files from ScanRefer download
```

## <a name="3"></a> 3. Data Preprocessing
First store the point cloud data for each scene into pth files.
```
python prepare_data.py
```
Split the files into train and val folders.
```
python split_train_val.py
```

## <a name="4"></a> 4. Pretrained Models
Please download the [*pretrained instance segmentation model*](https://www.dropbox.com/sh/u2mozpyzycwomwc/AABbYCbZPKGu8foT3bQc_jdna?dl=0) and place into the folder like this.
```
This Repository
|--GRU
    |--checkpoints
        |--model_insseg-000000512.pth
|--BERT
    |--checkpoints
        |--model_insseg-000000512.pth
```
Pretrained model for [*referring model with gru encoder*](https://www.dropbox.com/sh/u2mozpyzycwomwc/AABbYCbZPKGu8foT3bQc_jdna?dl=0) and place into the folder like this.
```
This Repository
|--GRU
    |--checkpoints
        |--gru
            |--models
                |--gru-000000032.pth
```
Pretrained model for [*referring model with bert encoder*](https://www.dropbox.com/sh/u2mozpyzycwomwc/AABbYCbZPKGu8foT3bQc_jdna?dl=0) and place into the folder like this.
```
This Repository
|--BERT
    |--checkpoints
        |--bert
            |--models
                |--bert-000000064.pth
```

## <a name="5"></a> 5. Training
Train the referring model with GRU encoder. (Note that we train with 2 GTX 1080Tis and Batchsize 8)
```
cd GRU/
python unet_gru_CoP.py
```


## <a name="6"></a> 6. Validation
Validate referring model with GRU encoder.
```
cd GRU/
python unet_gru_CoP_val.py
```
Validate referring model with BERT encoder.
```
cd BERT/
python unet_bert_val_CoP.py
```

## <a name="7"></a> 7. Acknowledgements

Our dataloader and training implementations are modified from https://github.com/facebookresearch/SparseConvNet and https://github.com/daveredrum/ScanRefer, please go check out their repositories for sparseconvolution and 3D referring object localization implementations respectively. We would also like to thank the teams behind Scannet and ScanRefer for providing their dataset.
