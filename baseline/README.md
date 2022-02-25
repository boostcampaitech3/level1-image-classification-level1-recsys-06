# pstage_01_image_classification


## Getting Started  
---
### Ground Rules

---
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
---
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
|resnext101_32x8d  |1|64| 47.51 |
|wide_resnet50_2   |1|64| 38.57 |
|wide_resnet101_2  |1|64| 38.23 | 0.0919|	30.1587|
|vgg11             |1|64| 52.33 |
|vgg11_bn          |1|64| 27.75 |
|vgg13             |1|64| 55.03 |
|vgg13_bn          |1|64| 28.36 |
|vgg16             |1|64| 57.14 | 0.4165|	58.1587|
|vgg16_bn          |1|64| 27.12 |
|vgg19             |1|64| 58.99 |
|vgg19_bn          |1|64| 27.59 | 0.0394|	18.1905|
|alexnet           |1|64| 43.23 | 0.1974|	41.1270|

- batchnorm 이 추가된 모델들이 성능이 좋지 않았다.






## EDA
---
<details>
<summary>Outliers</summary>
<div markdown="3">

|female -> male|male -> female|incorrect<-> normal|
|--------------|--------------|-------------------|
|<span style="color:yellow">000010</span>|001498-1|000020|
|<span style="color:yellow">000357</span>|004432|005227|
|<span style="color:yellow">000664</span>|005223|
|<span style="color:yellow">000667</span>|
|<span style="color:yellow">000725</span>|
|<span style="color:yellow">000736</span>|
|<span style="color:yellow">000767</span>|
|<span style="color:yellow">000817</span>|
|001720|
|<span style="color:yellow">003780</span>|
|<span style="color:yellow">003798</span>|
|<span style="color:yellow">004281</span>|
|006359|
|006360|
|006361|
|006362|
|006363|
|006364|
|<span style="color:yellow">006504</span>|

<span style="color:yellow">not sure</span>

</div>
</details>

<br>

## Data Augmentation
---
