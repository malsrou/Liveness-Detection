# Liveness-Detection
Code base for the liveness detection demo presented at Zaka Machine Learning Certification. 

The following Git Repository provides the code for the demo presented at the Zaka Machine Certification Capstone Project Presentations. We have used transfer learning to adapt the VGG19 model for liveness detection as well as a new MSIICNNet Model (Multi-Step Image Inferred Convolutional Neural Network) to detect Real or Fake for cropped faces and then we combined the predictions of both models to perform a global liveness detection predection on the frame stream from the web camera stream. 

It is important to note that the models are not trained on the whole dataset of CelebA-Spoof or ForgeryNet, in the upcoming update we will include the weights and model architecture of MSIICNet that have been trained on the full datasets. 

The weights for the VGG19 model and for the MSIICNNet can be found here: 
https://drive.google.com/drive/folders/1vjAhE78F6IZhv7vvw3uH4M5M0qHDKuB1?usp=sharing




