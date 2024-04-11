# **Spatially and Temporally Optimized Audio-Driven Talking Face Generation**
![pipeline03](https://github.com/donge1024/TalkingFace/assets/114487375/1b58d2ac-b59e-40a5-990c-92b53a197881)
This paper proposes to enhance the quality of generated talking faces  with a new spatio-temporal consistency. Specifically, the temporal consistency is achieved through consecutive audio frames of the same phoneme, which form temporal modules that exhibit similar lip appearance changes. This allows for adaptive adjustment in the lip movement for accurate synchronization.
The spatial consistency pertains to the uniform distribution of textures within local regions, which form spatial modules and regulate the texture distribution in the generator. This yields fine details in the reconstructed facial images. Extensive experiments show that our method can generate more natural talking faces than previous state-of-the-art methods in both accurate lip synchronization and realistic facial details.
## **Installation**
We train and test based on Python3.8. To install the dependencies run:

```conda env create -f environment.yml```

## **Test**
Prepare testing data：
prepare driving_audio
prepare driving_video 

```python inference.py --checkpoint_path --face --audio```

