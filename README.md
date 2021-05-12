# EmoRecCom
[ICDAR2021 Competition Multimodal Emotion Recognition on Comics scenes](https://sites.google.com/view/emotion-recognition-for-comics)

---

## Repository Details
- dataset.py - module to create dataset
- train.py - training module
- test.py - module to predict on the test data
- metric.py - custom metric function
- model.py - create model to fit data
---

## Setup and Install
- Install dependencies
```
pip install -r requirements.txt
```
- Download dataset
```
bash download_dataset.sh
```
---

### Training and Testing
```
python train.py
python test.py
```
Check the arguments of the module to try out various inputs

---

### Dataset details
- Note: participants are provided 6,112 training examples with the respective annotated labels. The testing set consists of 2046 examples without labels.
- Data format: 
  - `train_transcriptions.json`: contains auto-transcriptions in comic scenes
  - `train` contains 6,112 raw images of training data
  - `train_emotion_labels.csv`: contains binary labels
  - `additional_infor:emotion_polarity.csv`: contains additional info, the polarities of emotions in (0,1). Participants are encouraged to leverage this additional resources to achieve better performace.
  - `test`: contains 2,046 raw images of testing data
  - `test_transcriptions.json` contains auto-transcriptions in comic scenes
- Submission file:
  - File `results.zip` contains a submission sample. Please strictly follow the name convention and format of the csv file. Generally, your file needs to have 10 columns (without headers): id,image_id,angry,disgust,fear,happy,sad,surprise,neutral,other.
