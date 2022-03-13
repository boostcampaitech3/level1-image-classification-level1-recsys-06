# level1-image-classification-level1-recsys-06
# RecSys 6조

##### 강영석, 김예지, 박정규, 이호진, 홍수연

#### 1.프로젝트 개요
|목록|내용|
|--------------------------|------------|
|프로젝트 주제 |마스크 착용 상태 분류|
|프로젝트 개요 |마스크의 착용 상태(wear, incorrect wear, not wear), 성별(male, female), 연령대(30세 미만, 30세이상 60 세미만, 60세이상)를 결합하여 18 개 class를 구성하고, 이를 image classification합니다.|
|개발 환경| IDE : VSCode 협업 Tool : GitHub Library : pytorch, torchvision, pandas, matplotlib,nni, sklearn|

##### 프로젝트 구조 및 사용 데이터셋의 구조도
```
├── input// the data

│ ├── eval

│ │ ├──images

│ │ └──info.csv

│ └── train

│ ├──images

│ └──train.csv

├── baseline // here we create our models

│ ├── nni

│ ├── dataset.py

│ ├── earlystopping.py

│ ├── ensemble.py

│ ├── inference.py

│ ├── loss.py

│ ├── model.py

│ ├── requirements.txt

│ ├── README.md

│ ├── sample_submission.ipynb

│ └── train.py

├── EDA.ipynb

└── img_cl_mask.ipynb|
```
#### 2. 프로젝트 팀 구성 및 역할

* 강영석 : pretrained model추가, k-fold validation 적용, randaugment 적용
* 김예지 : augmentation(albermentation) 적용 및 loss 수정
* 박정규 : mixup 기법(target:60세 이상 데이터)을 사용한 augmentation 적용
* 이호진 : pretrained model 추가, ensemble 추가, early stopping 추가, nni를 사용한 auto ml 추가
* 홍수연 : preprocessing, augmentation(data generation)


#### 3. 프로젝트 수행 절차 및 방법

* 프로젝트를 진행하기 위한 환경을 세팅하고, EDA, Augmentation을 거친 후, model을 선정하여 Hyperparameter를 수정해가며 성능 비교
![table](/img/table.png)

#### 4. 프로젝트 수행 결과

##### EDA
![EDA1](/img/EDA1.png)
![EDA2](/img/EDA2.png) ![EDA](/img/EDA3.png)

- 성비 : 여자 60%, 남자 40%로 여성의 비율이 약간 높음
- 비율이 낮은 연령대 : 전체적인 연령대 비율은 30~45세 구간의 데이터 분포가 낮음
- 비율이 높은 연령대 : 50~60세 구간의 데이터 분포가 가장 높음
- 클래스로 구분했을 때 50~60세의 비율 덕분에 `>= 30 and < 60` 클래스의 비율이 높음


##### Preprocessing, Data Generation
![Preprocessing2](/img/Preprocessing,DataGeneration2.png)

- Preprocessing : CenterCrop, Normalize 적용
- Augmentation : training 영상에 flip, jitter, Gaussian noise를 random하게 적용
- Data Generation : 60세 이상 label 추출하여 변형 영상 생성 및 데이터셋 추가
- albermentation 라이브러리 적용
```
CoarseDropout, RandomBrightnessContrast, GridDistortion, OpticalDistortion, ChannelShuffle, Gaus
sNoise, GaussianBlur, GlassBlur, MedianBlur, MotionBlur, RGBShift
```
- mixup(data augmentation) : 60세 이상 사진을 다른 class의 사진과 mix

##### K-fold validation 적용
- 원하는 개수 만큼 split해서 train과 validation을 나눠서 학습시킬 수 있도록 구현

##### Ensemble
- 2~4개의 모델들을 합쳐서 학습할 수 있도록 구현 & 한개의 모델을 여러 개를 돌린 결과를 soft voting 방식으로 구현

##### Early stopping
- 주어진 patience 만큼 val loss 가 증가 하지 않는 다면 과적합이 예상됨으로 학습을 중지 시킬 수 있도록 구현

##### nni Auto
- nni를 사용하여 자동적으로 최적의 param을 찾는 실험을 간편하게 할 수 있도록 함

##### Loss 수정

- CrossEntropyLoss에 weight를 적용
- Class-balanced Loss, Focal Loss 적용


#### ○ 최종 모델 선정 및 분석

1. 아키텍처 : Resnet34 (pre-trained model)
2. LB 점수(최종) : f1 score – 0.7097, accuracy – 78.
3. Data preprocessing : HorizontalFlip
4. img_size = 512 X 384
5. 추가 시도 : CrossEntropyLoss에 weight 적용, 모든 데이터를 train data로 활용

