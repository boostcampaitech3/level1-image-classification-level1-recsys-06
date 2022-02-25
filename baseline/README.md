# pstage_01_image_classification

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
