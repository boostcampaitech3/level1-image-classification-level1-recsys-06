# pstage_01_image_classification


## Getting Started  

### Ground Rules

일 10회 => 하루 인당 2회 
11시 30 이후 제출 기회 남았을 시 물어보고 반대하는 사람 없으면 제출가능.
or 오늘 제출 안한다고 미리 말하기

### Dependencies
> torch==1.6.0 <br>
> torchvision==0.7.0                                                              

### Install Requirements
```python
pip install -r requirements.txt
```


### Training
```python 
SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] NAME=[model name] epochs= [epochs] python train.py
```
 
### Inference
```python
SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] NAME=[model name] python inference.py
```
### Evaluation
```python
SM_GROUND_TRUTH_DIR=[GT dir] SM_OUTPUT_DATA_DIR=[inference output dir] python evaluation.py

```


## Pretrained Models 

<details>
<summary>Available Models</summary>
<div markdown="1">
<br>

- resnet18
- resnet34
- resnet50
- resnet101
- resnet152
- resnext50_32x4d
- resnext101_32x8d
- wide_resnet50_2
- wide_resnet101_2
- vgg11
- vgg11_bn
- vgg13
- vgg13_bn
- vgg16
- vgg16_bn
- vgg19
- vgg19_bn
- alexnet
- densenet121
- densenet161
- densenet169
- densenet201
- efficientnet_b0
- efficientnet_b1
- efficientnet_b2
- efficientnet_b3
- efficientnet_b4
- efficientnet_b5
- efficientnet_b6
- efficientnet_b7
- googlenet

</div>
</details>
<br>

<details>
<summary>Baseline params</summary>
<div markdown="2">
<br>

> epoch:<br>
 batch_size:<br>
 etc:
 
</div>
</details>
<br>

|Model|epoch|batch_size|val_acc|board_f1|board_acc|
|-----|-----|----------|-------|------------|----|
|resnet18          |1|64| 42.35 |
|resnet34          |1|64| 42.43 | 
|resnet50          |1|64| 40.79 |
|resnet101         |1|64| 43.94 |
|resnet152         |1|64| 42.25 | 0.1322|	36.5556|
|resnext50_32x4d   |1|64| 42.12 |
|resnext101_32x8d  |1|64| 47.51 | 0.1815|	41.7460|
|wide_resnet50_2   |1|64| 38.57 |
|wide_resnet101_2  |1|64| 38.23 | 0.0919|	30.1587|
|vgg11             |1|64| 52.33 |
|vgg11_bn          |1|64| 27.75 |
|vgg13             |1|64| 55.03 |
|vgg13_bn          |1|64| 28.36 |
|vgg16             |1|64| 57.14 | 0.4165|	58.1587|
|vgg16_bn          |1|64| 27.12 |
|vgg19             |1|64| 58.99 | 0.4325|	58.7937|
|vgg19_bn          |1|64| 27.59 | 0.0394|	18.1905|
|alexnet           |1|64| 43.23 | 0.1974|	41.1270|
|densenet121       |1|64| 46.11 |
|densenet161       |1|64| 43.49 |
|densenet169       |1|64| 46.46 |
|densenet201       |1|64| 46.64 |
|efficientnet_b0   |1|64| 27.59 |
|efficientnet_b1   |1|64| 20.40 |
|efficientnet_b2   |1|64| 19.63 |
|efficientnet_b3   |1|64| 22.12 |
|efficientnet_b4   |1|64| 16.02 |
|efficientnet_b5   |1|64| 13.78 |
|efficientnet_b6   |1|64| 9.42  |
|efficientnet_b7   |1|64| 11.69 |
|googlenet         |1|64| 28.68 |


- batch norm 이 추가된 모델들이 성능이 좋지 않았다.





## EDA

<details>
<summary>Outliers</summary>
<div markdown="3">

|female -> male|male -> female|incorrect<-> normal|
|--------------|--------------|-------------------|
|000010 💥|001498-1|000020|
|000357 💥|004432|005227|
|000664 💥|005223|
|000667 💥|
|000725 💥|
|000736 💥|
|000767 💥|
|000817 💥|
|001720|
|003780 💥|
|003798 💥|
|004281 💥|
|006359|
|006360|
|006361|
|006362|
|006363|
|006364|
|006504 💥|

💥 => not sure

 
</div>
</details>

<br>

## Data Augmentation
---
