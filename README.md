# Real-Time Face Recognition 

<br/>
<p align="center">
  <a href="https://github.com/elifezgisen/face-recognition">
    <img src="https://venturebeat.com/wp-content/uploads/2021/06/GettyImages-1195209545.jpg?fit=750%2C386&strip=all" alt="Logo" width="570" height="300">
  </a>

  <h3 align="center">Real-Time Face Recognition
</h3>

  <p align="center">
    <a href="https://github.com/elifezgisen/face-recognition"><strong>Explore the docs »</strong></a>
    <br/>
    <br/>
    <a href="https://github.com/elifezgisen/face-recognition">View Demo</a>
    .
    <a href="https://github.com/elifezgisen/face-recognition/issues">Report Bug</a>
    .
    <a href="https://github.com/elifezgisen/face-recognition/issues">Request Feature</a>
  </p>
</p>

![Downloads](https://img.shields.io/github/downloads/elifezgisen/face-recognition/total) ![Contributors](https://img.shields.io/github/contributors/elifezgisen/face-recognition?color=dark-green) ![Issues](https://img.shields.io/github/issues/elifezgisen/face-recognition) ![License](https://img.shields.io/github/license/elifezgisen/face-recognition) 

## Table Of Contents

* [About the Project](#about-the-project)
* [Technical Part](#technical-part)
* [Test Part](#test-part)
  * [Accuracy Rate Values](#accuracy-rate-values)
  * [Confusion Matrix Results](#confusion-matrix-results)

## About The Project

Face Recognition is the process of identifying a person or persons using their physical or characteristic features. Facial features, which are one of the biometric features in the field of information security, such as palm shape, fingerprint, iris, retina and voice
carried out using

This project has been prepared over OpenCV and Face Recognition libraries using Image Processing technology. Visual Studio Code IDE and Python programming language were used in the code phase.

The Face Recognition module uses the "Dlib" library in the background. The Dlib library creates the "ResNet-34" model, which is an Artificial Neural Network model. This is a "CNN (Convolutional Neural Network)" architecture.

## Technical Part

- The Face Recognition module uses a technique called "Deep Metric Learning" to perform Face Recognition. This technique attempts to measure similarity between samples while executing learning tasks using an appropriate distance metric. Euclidean distance is used in this application.

- Instead of outputting a single label (coordinates of objects in the image/bounding box), this method outputs a real-valued feature vector.

- Dlib, the output vector for the Face Recognition network, consists of a list of 128 real-valued numbers used to measure the face. This is how neural network training is done.

- The network architecture used for the face recognition application is based on the ResNet-34 model created by the Dlib library.
The network itself has been trained by "Davis King" over a dataset of approximately 3 million images.

- The output layer of the model has 128 nodes. So any face
When you feed the image to the ResNet model as data, it produces a 128-dimensional vector.

- After creation of the input vector, the two face images
are compared and both are fed as data to the ResNet model, respectively.
Thus, two 128-dimensional vectors are obtained as output.

- Finally, the similarity of these two vectors is checked. Cosine Similarity and Euclidean Distance are the most common methods used to find similarity between two vectors. The adjusted threshold value for the Euclidean Distance was determined as 0.6.

- While determining this threshold value, first positive and negative identity pairs are passed to the ResNet model and their representations are found. Then, the Euclidean distance is calculated for each even vector. The lower the threshold value, the more stringent the facial recognition system.

- When comparing two people, after the value of the distance is determined by Euclidean Distance or Cosine Similarity, this value is small for the same people and large for different people. The model is trained in this way.

## Test Part

- A readily available data set was used for training data in order not to violate data privacy. This open-source dataset named “Georgia Tech” contains images of 50 people in total. Each person in the dataset has 15 color JPEG images with a complex background, taken at a resolution of 640x480 pixels.

- For the project, 10 images of 10 people in total were taken from the data set and named on the file. The remaining 5 images were set aside for later use in the test section.

### Accuracy Rate Values:


True Positive (TP) = Recognized faces from dataset: 100/100

False Negative (FN) = Unrecognized faces from dataset: 0

True Negative (TN) = Correctly detected faces that are not in the dataset (Unknown): 40/100

False Positive (FP) = Incorrectly detected faces that are not in the dataset (Different people from the dataset): 60/100


### Confusion Matrix Results:

**Accuracy** = (TP + TN) / (TP + TN + FP + FN)

           = (1 + 0,4) / (1 + 0,4 + 0,6 + 0) = 0,7
         
           = %70
