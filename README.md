
# MTCNN Keras Implementation, suitable for memory & battery constrained device
Keras Implementation of Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks.This is a Keras implementation (Python) of a tweaked version of MTCNN for suitable MCU deployment
The project implemenents MTCNN in Keras with a slight variation in the model architectures of Pnet, Rnet and Onet.
The model architecture is tweaked to reduce the number of activations, weights size and trainable parameters. This is a pipeline that is focused on minimizing RAM and ROM, in order to deploy the Keras Implementation of MTCNN in a battery and memory contrained device. It could easily be deployed in RasberryPi, ARM based MCU, GAP8 based MCU. Suggested MCU could be STM32L475VGX based on ARM Cortex M4. Models are deployable in C through CubeAI, however this repository refers only to the Python Implementation of MTCNN.


# Important Information
Model is initially trained on WiderFace Dataset for Bounding Box Regression
Model is also trained on CelebA Dataset for facial landmark localization
Model is evaluated on the custom made Dataset Politcs 101
Model is fully deployable to a memory constrained MCU (it can meet the memory requirements of 1MB FLASH and 128KB RAM)

# Politics 101
In this project we introduce a custom made dataset (made using Makesense.ai) called Politics 101. The dataset consists of 10 classes of famous politiical leaders. The images have a low resolution, to pose an extra challenge during the evaluation of the face detection system. The dataset focuses mostly on face detection but has been built with face classification/recognition/identification in mind. All images are savd under "image.npy" at a (320,320,3) input size and the corresponding labels are saved under "labels.npy". Further information are included in the datasets corresponding ReadMe file.


# Refined training process
The new training strategy also follows the original caffe code. i.e. randomly select Classification loss, roi regression loss or key point regression losses and minimize it for each batch of data. 

# Model Evaluation
The pipeline is evaluated using the Politcs 101 dataset and we view evaluation as both an imbalanced classification task, as well as a regression task, to improve bounding box predictions. The selected IoU is 0.5, but experimenting with teh model IoU and the pipeline IoU can affect greatly the perfomance of the pipeline. Also depending on the input dimension of the Pnet , as well as the output layers of Onet, performance varies greatly.


**Credits** to **xiangrufan** for providing the pretrained weights for the Pnet, Rnet and Onet, as well as the initial Keras Implementation of MTCNN pipeline. **Transplanted** from **https://github.com/xiangrufan/keras-mtcnn**


