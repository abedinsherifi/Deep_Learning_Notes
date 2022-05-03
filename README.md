![Your Repository's Stats](https://github-readme-stats.vercel.app/api?username=abedinsherifi&show_icons=true)
![Your Repository's Stats](https://github-readme-stats.vercel.app/api/top-langs/?username=abedinsherifi&theme=blue-green)
![](https://komarev.com/ghpvc/?username=abedinsherifi)

<p align="center">
  <a href="https://github.com/prespafree1/Deep_Learning_Notes">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/prespafree1/Deep_Learning_Notes.svg">
  </a>
  <a href="https://github.com/prespafree1/Deep_Learning_Notes">
    <img alt="GitHub forks" src="https://img.shields.io/github/forks/prespafree1/Deep_Learning_Notes.svg">
  </a>
    <a href="https://github.com/prespafree1/Deep_Learning_Notes/graphs/contributors" alt="Contributors">
        <img src="https://img.shields.io/github/contributors/prespafree1/Deep_Learning_Notes" /></a>
</p>

# Learning Affordance for Direct Perception in Autonomous Driving by Chenyi Chen, et al
# Paper Summary

### Vision based autonomous driving:
-> mediated perception approach – parse entire frame in order to make driving decision
-> behavior reflex approach – directly map input image to a driving action by a regressor

Direct perception method developed by this paper.
Trained a deep CNN using a recording from 12 hours of human driving in a video game. 
Trained model in car distance estimation using KITTI dataset.

**5 convnet layers followed by 4 fully connected layers (4096, 4096, 256, and 13 for output dimensions). 
Screenshots are down-sampled to 280 x 210.** 

They collected **484,815 images for training.** 

Initial **learning rate** of 0.01

**mini batch of 64 images randomly selected** from the training samples

after 140,000 iterations they stopped the training process. 

7 different tracks and 22 different cars used

**The convnet processes the TORCS images and estimates the 13 indicators for driving. Based on indicators and speed of car, controller will send commands to the car.**

**13 affordance indicators:**
angle
toMarking_LL
toMarking_ML
toMarking_MR
toMarking_RR
dist_LL
dist_MM
dist_RR
toMarking_L
toMarking_M
toMarking_R
dist_L
dist_R


# End to End Learning for Self-Driving Cars by NVIDIA
# Paper Summary

**Trained a CNN to map raw pixels from a single camera image directly to steering commands. 
System operating at 30 FPS.**

CNNs widely used due to Large Scale Visual Recognition Challenge (ILSVRC) dataset for training and testing and due to GPUs. 

**CNN Layout:**
Input Plane (3@66x200)
Normalization
Normalized input plane (3@66x200)
Convolutional feature map (24@31x98)
Convolutional feature map (36@14x47)
Convolutional feature map (48@5x22)
Convolutional feature map (64@3x20)
Convolutional feature map (64@1x18)
Flatten
1164 neurons dense layer
100 neurons dense layer
50 neurons dense layer
10 neurons dense layer
outpur (vehicle control)

**convo layers perform feature extraction. They have different kernel sizes either 3x3 or 5x5 in this case.** 

**In data augmentation, the set of frames are augmented by adding artificial shifts and rotations.** 

No labels were used for outline of road etc. The steering commands sent out from the network were 1/r, where r is the the turn radius. 


# Deep Learning Notes

**Regression** is when a model, such as a neural network, accepts input and produces a numeric output.
The output of a **classification** model is what class the input belongs to.

**In deep learning, the more data present the higher the performance.**

CNNs can scan an  image for patterns within the image. 
Recurrent NN can find patterns across several inputs, not just within a single input. 

**Frameworks for deep learning**: TensorFlow (Google), MXNet (Amazon), Theano (Univ of Montreal), and CNTK (Microsoft). 

Keras – is a high-level neural network API, written in Python and able to run on top of TensorFlow, CNTK, or Theano


Data fed into a machine learning model needs to be normalized. **Zscore** used for normalization. 

Z = input – mean / standard deviation. 

Training/Validation split (80/20) or K-Fold Cross Validation

Activation functions, also known as transfer functions, are used to calculate the output of each layer of a neural network. 

**ReLU (Rectified Linear Unit) used for output of hidden layers. 
Softmax used for the output of classification neural networks. 
Linear used for the output of regression neural networks.**

ReLU = max(0,𝑥)


**hidden layer values = A (W1 * x + b1), where A is the activation function, x is the input, W1 is the weight, and b1 is the bias.** 

Good idea to save big neural networks so they can be reloaded later. A reloaded nn will not require training. NN can be saved as YAML (just structure no weights) or JSON (just structure no weights) or HDF5 (structure + weights). 

**Overfitting** occurs when a neural network is trained to the point that it begins to memorize rather than generalize. 

The **mean square error** is the sum of the squared differences between the prediction (𝑦̂ ) and the expected (𝑦). MSE values are not of a particular unit. If an MSE value has decreased for a model, that is good. However, beyond this, there is not much more you can determine. Low MSE values are desired.

The **root mean square (RMSE)** is essentially the square root of the MSE. Because of this, the RMSE error is in the same units as the training data outcome. Low RMSE values are desired.

https://abedinsherifi.github.io/Deep_Learning_Notes/
