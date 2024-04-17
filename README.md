# **Spatially and Temporally Optimized Audio-Driven Talking Face Generation**
![pipeline03](https://github.com/donge1024/TalkingFace/assets/114487375/1b58d2ac-b59e-40a5-990c-92b53a197881)
This paper proposes to enhance the quality of generated talking faces  with a new spatio-temporal consistency. Specifically, the temporal consistency is achieved through consecutive audio frames of the same phoneme, which form temporal modules that exhibit similar lip appearance changes. This allows for adaptive adjustment in the lip movement for accurate synchronization.
The spatial consistency pertains to the uniform distribution of textures within local regions, which form spatial modules and regulate the texture distribution in the generator. This yields fine details in the reconstructed facial images. Extensive experiments show that our method can generate more natural talking faces than previous state-of-the-art methods in both accurate lip synchronization and realistic facial details.

## **Highlights**
-  temporal consistency based on consecutive frames of the same phoneme, which enables adaptive lip movement over time for better lip synchronization
-  spatial consistency based on local regions of consistent texture distribution, which constrains the generator to output facial images with fine details

## **Installation**
-  We train and test based on Python 3.8
-  ffmpeg: ```sudo apt-get install ffmpeg```
-  To install the dependencies run: ```conda env create -f environment.yml```


## **Test**
- Temporal module processing：

  ```python temporal_processing.py```

  **Forced Alignment**:First, simulate the process of forced alignment from the input audio data to obtain the phoneme sequence extracted from the speech.

  **Construct Temporal Module**:Construct the corresponding temporal module based on the obtained phoneme sequence.

  **Calculate Lip Loss**:For each temporal module, calculate lip loss.

  **Update Lip Weight**:Update the lip weight based on the lip movement similarity in the temporal module.

- Spatial module processing：

  ```python spatial_processing.py```

  **Texture Block Calculations**:First, perform texture block calculations on the image and divide the image into multiple blocks for subsequent processing.

  **Calculate Distance**:Calculate the distance between each pixel and the cluster center of its surrounding area.

  **Calculate Spatial Module**:Determine the spatial module in the image based on the distance calculation results.

  **Face Optimization**:Use the information of the spatial module to optimize the generated face image.

  **Note**:The audio used in our work should be sampled at 16,000 Hz and the corresponding video should have a frame rate of 25 fps.

- Run the demo：

  ```python inference.py --checkpoint_path checkpoints/ --face input/.video --audio input/.wav```

  The result is saved in results/result.mp4.

## **Contact**
Our code is for research purposes only. More details will be released shortly. If you have any questions, please contact us: biaodong@bit.edu.cn
