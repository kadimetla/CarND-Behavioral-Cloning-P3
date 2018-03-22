# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

model layers:
The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Input Normalization    | (pixel_value/255.0) - 0.5           |
| Image Cropping    | 70 rows pixels from the top of the image <br> 25 rows pixels from the bottom of the image <br> 0 columns of pixels from the left of the image <br> 0 columns of pixels from the right of the image <br> 65x320x3 RGB cropped image |
| Convolution 5x5 | stride 2x2 output 24 channels | 
| Activation      | relu                          |
| Convolution 5x5 | stride 2x2 output 36 channels | 
| Activation      | relu                          |
| Convolution 5x5 | stride 2x2 output 48 channels | 
| Activation      | relu  
| Convolution 3x3 | stride 1x1 output 64 channels | 
| Activation      | relu  |
| Convolution 3x3 | stride 1x1 output 64 channels | 
| Activation      | relu  |
| Flatten      | flatten  |
| Dense        | dense 100 |
| Dense        | dense 50 |
| Dense        | dense 10 |
| Dense        | dense 1 |

Compile model using mean square error loss function and adam optimizer.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road by driving on track 1 and 2. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to minimize the mean-squared error between the steering command output by the network

My first step was to use a convolution neural network model similar to the Nvidia Network Architecute I thought this model might be appropriate because it trains the weights of our network to minimize the mean-squared error between the steering command output by the network

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model in the augmented data for the flipped images the steering angle is multiplied by -1.5 and number of epohs to 2.

Then I kept tuning the network until it was able to complete the rounds on track 1.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and to improve the driving behavior in these cases, I trained the model in track 2 and in the augmented data for the flipped images steering angle is multiplied by -1.5.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers as shown in the table above.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![center](./samples/center_2018_02_12_20_22_50_040.jpg?raw=true)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive in the middle of the road. These images show what a recovery looks like starting from ... :

![left](./samples/left_2018_02_12_20_22_50_040.jpg?raw=true)
![right](./samples/right_2018_02_12_20_22_50_040.jpg?raw=true)


Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would help. For example, here is an image that has then been flipped: I have not captured flipped images.
<!--
![alt text][image6]
![alt text][image7]
-->

After the collection process, I had 10516 number of data points. I then preprocessed this data by collecting center images and steering angle for each input.


I finally randomly shuffled the data set using 80% training and 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
