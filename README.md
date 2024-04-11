# **Spatially and Temporally Optimized Audio-Driven Talking Face Generation**
![pipeline03](https://github.com/donge1024/TalkingFace/assets/114487375/1b58d2ac-b59e-40a5-990c-92b53a197881)
## **Installation**
We train and test based on Python3.8. To install the dependencies run:

```conda env create -f environment.yml```

## **Test**
Prepare testing data：
prepare driving_audio
prepare driving_video 

```python inference.py --checkpoint_path --face --audio```

