# Pretrained CNN models for Pytorch (Work in progress)

The goal of this repo is:

+ We provide pre-trained CNN models to help bootstrap computer vision applications. (Res50 Top1 larger than 77 or even better)

## Models

| Model          | Top 1/5 (ours) | Top 1/5 (Paper) | Model |
| :------------: | :------------: | :-------------: | :---------------: |
| Resnet 50      | 78.83/94.23    | 75.3/92.2       |  [Model](https://drive.google.com/file/d/1ruQq4noE8Y50hocgJXJ9spDNNtX6ujrG/view?usp=sharing)| 


## Evaluation

```shell
python main.py VAL_DIR -e --load_ckpt res50.pth
```

## News

+ 24/12/2018 Add Resnet50
+ 29/09/2018 First Init