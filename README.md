# Preparation
## environment
```shell
pip install -r requirements.txt
```
## dataset
The data is already in this repo, to learn more you can go [`CUB-200-2011`](https://data.caltech.edu/records/65de6-vp158)
It is recommaned to put the data file in the dataset folder like
```
dataset/
    attributes.txt
    CUB-200-2011/
        images/
        ...
```
```
python split.py # Run this command when the data is ready, which is to partition the data set
```

# Train & Evaluation
There are a number of options that can be set, most of which can be used by default, which you can view in `train.py`.
## for train
```
python train.py --pretrained # if you want to finetune the pretrained model
```


## for evaluation
```
python eval.py --evaluate --file <model_ckp_path>
```
The weights after model training can be downloaded [`here`](https://drive.google.com/drive/folders/1pNmBDDdkkK1vl0aWQfTnBCmqp0USbD6C?usp=drive_link)


## for hyper parameters(learning rate) reseaech
`hyper_param_search.py` is used to search for appropriate learning rate settings, you can use it directly but it may consume much time.
