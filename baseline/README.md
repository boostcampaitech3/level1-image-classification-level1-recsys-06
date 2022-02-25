# pstage_01_image_classification

<<<<<<< HEAD
<<<<<<< HEAD
## Getting Started    
### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py`
SM_CHANNEL_TRAIN=/opt/ml/input/data/train/images SM_MODEL_DIR=/opt/ml python train.py
### Inference
- `SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py`
SM_CHANNEL_EVAL=/opt/ml/input/data/eval SM_CHANNEL_MODEL=/opt/ml/exp SM_OUTPUT_DATA_DIR=/opt/ml/result python inference.py

### Evaluation
- `SM_GROUND_TRUTH_DIR=[GT dir] SM_OUTPUT_DATA_DIR=[inference output dir] python evaluation.py`
=======
=======
>>>>>>> 9841b696ea52e4cee307cf8b1b7d9252c8864fa3

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
- resnet152
- vgg19
- alexnet
- densenet161
- googlebet
- efficientnet_b0
- efficientnet_b4
- efficientnet_b7
- vit (resize to 224 224)
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

|Model|epoch|batch_size|baseline|leaderboard|
|-----|-----|----------|-------|------------|
|resnet152       |20|64| 71.93 |
|vgg19           |20|64| 73.33 |62.8095|
|densenet161     |20|64| 72.94 |
|alexnet         |20|64| 68.36 |47.9841|
|googlenet       |20|64| 66.77 |46.9683|
|efficientnet_b0 |20|64| 60.63 |
|efficientnet_b4 |20|64| 55.58 |
|efficientnet_b7 |20|64| 54.05 |33.8889|
|vit             |20|64| 74.55 |
* used [128,96] resize except vit(used [224,224] resize )





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
<<<<<<< HEAD
>>>>>>> 9841b696ea52e4cee307cf8b1b7d9252c8864fa3
=======
>>>>>>> 9841b696ea52e4cee307cf8b1b7d9252c8864fa3
