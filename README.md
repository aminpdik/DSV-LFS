


## Installation
```
Check requirements.txt file for packages
```

## &#x1F527;Get Started
**Just follow these steps to train and test DSV-LFS.**
### Dataset 
**1.** Download the dataset from the following links.
+ PASCAL-5<sup>i</sup>: [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) + [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)
+ COCO-20<sup>i</sup>: [MSCOCO2014](https://cocodataset.org/#download)

#### SAM ViT-H weights
Download SAM ViT-H pre-trained weights from the [link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

#### llava-v1.5-7b
Download llava-v1.5-7b modek from the [link](https://huggingface.co/liuhaotian/llava-v1.5-7b).

### Training
```
deepspeed  train_ds.py \
  --version="PATH_TO_llava-v1.5-7b" \
  --dataset_dir='./dataset' \
  --vision_pretrained="PATH_TO_SAM" \
  --benchmark= "pascal" or "coco" \
  --fold="0" \
  --exp_name="name"\
  --shot="1" \
```
### Inference
```
deepspeed  train_ds.py \
  --version="PATH_TO_llava-v1.5-7b" \
  --dataset_dir='./dataset' \
  --vision_pretrained="PATH_TO_SAM" \
  --benchmark= "pascal" or "coco" \
  --fold="0" \
  --exp_name="name"\
  --shot="1" or "5" \
  --eval_only="True" \
```

