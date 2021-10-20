# Designing, Building, Testing and Critiquing a System for Performing Face Alignment.

## Designing the System

We were tasked with creating a system to perform the face recognition task. We designed two different systems to try and perform this task based on a series of different papers (Ciao, 2013) (Kazemi, 2014) we read on the topic, but eventually decided on the simpler cascaded regression approach.

```
for K regressors
    for N images
        if lastPrediction == Null
            lastPrediction = averagePoints
        calculate image sift features A based on lastPrediction
        calculate targets for image N as P
    train sklearn linear regression model
    update lastPrediction on K and dampingFactor
```
The pseudocode above, as adapted from the specification, describes the process of training for our series of cascaded regressor. K refers to the number of regressors that we want to use, and N is the full list of images. Since our data has already been resized to all be the same height and width, and the eyes have been mostly aligned, the first step towards getting our algorithm to work is creating a starting point for the predictions. The pre-processing of the images before we received them makes the creation of an average face a lot easier, and this works well as a starting point. 

Numpy has a built in function for finding the averages of values, so creating a list of average points is as easy as iterating through each point and finding the averages of the x and y coordinates across all images. This average face is as good as we can get for a starting point. Running this average in comparison to the ground points for the training images, we get an average error of 10 pixels per point. While this cumulatively is a large error, it is already quite close to the correct solution. 

<p align="center">
   <img width="150" height="150" src="https://user-images.githubusercontent.com/92593423/138099182-40620b83-468c-447b-934b-0f0e950ad941.png">

  <br>
  <i> Figure 1: The average face generated from all images in the training set, <br>
    and the average points generated from all points in the training set. </i>
</p>
  
We decided to use SciKit-Learn to train our linear regressor. SKLearn is open-source and has some very good functions that allow easy implementation of various Machine Learning systems (Hao, 2019) including a linear regression model. The model minimises the sum of squares between two datasets – in this case the predicted points and the ground points. Our predicted points are sent in the form of SIFT descriptors. 

OpenCV has implementation for SIFT feature calculation and computation, so we can use this for our features. SIFT features contain both a key point (to mark a location) and a descriptor (to mark a direction), but we only need the direction. The direction of the feature is what we will send to the linear model to train or test it. 

## Pre-Processing

Pre-processing the images is a very important step towards a well-trained linear regressor. When the SIFT features are calculated, the image is searched for key points and it searches the pixels neighbouring this point in a certain radius to calculate the direction. Without pre-processing, the directions could come out wrong. 
The first pre-processing step we ran was making the images greyscale. This reduces the dimensionality of the images from 3 (one layer for each; red, green, and blue) to 1 (having just a single layer for pixel intensities). As shown in Figure 2, the image loses all colour, but is not altered in any other way.

<p align="center">
   <img width="300" height="150" src="https://user-images.githubusercontent.com/92593423/138100535-04c3b6fc-c82b-41d0-b4c6-5106cffbb2ef.png">
  <br>
  <i> Figure 2: A comparison between an initial image from the <br>
    example dataset and a greyscale version of the same image </i>
</p>

The next step of pre-processing was to add a Gaussian blur to the image. 
A Gaussian blur compares a pixel to its surrounding pixels and finds the average of them. This effect is more noticeable in coloured images, where transitions between coloured sections become a lot smoother, but on a greyscale image it mostly helps with reducing noise. Having too much noise in an image can cause anomalies within the SIFT descriptors, resulting in incorrect training data. The effects of the Gaussian blur on the example image are shown in figure 3.

<p align="center">
   <img width="150" height="150" src="https://user-images.githubusercontent.com/92593423/138100554-3a423dd5-29cf-4989-aafb-d9c9b67be86b.png">
  <br>
  <i> Figure 3: The greyscale image from figure 2, with the addition <br>
  of a gaussian blur. The blur has a block size of 5x5 pixels.</i>
</p>

To finish the pre-processing, we used OpenCV’s normalise and resize functions to remove more noise from the image and reduce its dimensionality further, as seen in figure 4. With a smaller image, the algorithm will run faster and have a smaller margin for error.

<p align="center">
   <img width="150" height="150" src="https://user-images.githubusercontent.com/92593423/138100582-eafa4011-886a-4141-b3c8-8a713ff9cb5a.png">
  <br>
  <i> Figure 4: The nromalised and resized version of the image <br>
    shown above in figure 3.</i>
</p>

Lastly, before setting the system to train, we took the last 100 images from the dataset to use as validation images. This allowed us to test the system on data the regressor had not yet seen, giving us an accurate and quantitative test, as we were able to compare the points to correct ground points.

## Training the System

Following the Pseudocode shown in the _designing the system_ section, we were able to construct the cascaded regressor and set it to train on the data. The source code can be viewed in the attached colab notebook. 

## Testing the System
Using the validation set, we noticed a lot of small issues with the solution that we spent time perfecting. We performed a series of experiments to see the effect of different effects. The first experiment we performed looked at different amounts of regressors and damping factors. Shown in the graph (figure 5) there was a large trough at a damping factor of 0.15 and 25 regressors. Running the training and testing with the validation set with these values, we got significantly better results.

<p align="center">
   <img width="200" height="150" src="https://user-images.githubusercontent.com/92593423/138101473-99b155a0-1ed1-4c02-978a-a4208950668b.png">
  <br>
  <i> Figure 5: A graph showing the error amount across a series of <br>
    different damping factors, and a varying number of regressors </i>
</p>

We tinkered further and noticed that normalisation and converting the images to greyscale also make the system less accurate and robust. This is most likely because these functions both reduce the range of pixel intensities in the image, meaning sift features were struggling to find accurate descriptors.

<p align="center">
   <img width="450" height="300" src="https://user-images.githubusercontent.com/92593423/138101922-ba61e9b2-1826-4df8-9324-890bae34c4c2.png">
  <br>
  <i> Figure 6: A Demonstration of the accuracy of the system over a <br>
  selection of face shapes, angles, rotations, ages, and light levels.</i>
</p>

## Failure Cases

Some images in our example and validation data resulted in incorrect predictions, but this was typically only on the images where the picture contained large amounts of occlusion or where the pose wasn’t natural. As shown in figure 7, the algorithm struggled with these two images in particular, getting confused with the mouths at their odd.
<p align="center">
   <img width="300" height="150" src="https://user-images.githubusercontent.com/92593423/138102093-761ef5f4-b1a5-4cad-94f8-d3542c9a50c5.png">
  <br>
  <i> Figure 7: Two images from the validation set which demonstrate <br>
    situations where the system may struggle to find landmarks if <br>
    strange faces are being made. </i>
</p>

## Graphical Effects

Adding graphical effects to the images was a relatively simple task. While OpenCV does have functions to superimpose images onto another, it wasn’t as dynamic as we would have liked, so we imported the PIL Image libraries. This adds a selection functions that allow us to easily superimpose, adjust, move, and rotate images so that they look good. 

We decided to add the flower crown filter, made famous by applications such as snapchat and instagram, to the images as it is a simple and well-known effect. To figure out the location the crown needs to go, we calculated the coordinates of points 0 and 6 (marking the temples) to calculate the width the flower crown needed to be, and used the coordinates of points 8 and 11 (marking the peak of the eyebrows) to calculate the angle of the crown and the coordinates of how far up the forehead it needs to rest. 

<p align="center">
   <img width="150" height="150" src="https://user-images.githubusercontent.com/92593423/138102601-aa7a5913-763b-40be-8003-d36e12257855.png">
  <br>
  <i> Figure 8: The average face and points, marked with <br>
  numerical values to deonate their position in a list. <br>
  src: I. Simpson: Computer Vision Coursework Assignment - Spring 2021 <br>
  University of Sussex, canvas.sussex.ac.uk </i>
</p>



Shown in figure 9 are 4 different example images with the flower crowns added. Images 9A and 9C demonstrate that the algorithm functions correctly, even with the face being rotated. Image 9D shows that having the face angled slightly away from the camera can make the image look a bit off, but it doesn’t look bad.

<p align="center">
   <img width="300" height="300" src="https://user-images.githubusercontent.com/92593423/138103348-74622e38-3afa-4e61-b5f2-7c0b809a59ff.png">
  <br>
  <i> Figure 9: 4 of the 6 given example images with flower <br>
  crowns added, demosntrating the algorithm is reobust enough <br>
  to support at least a few different angles of rotation, face <br>
  shapes, and light levels.</i>
</p>


