##Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/timcamber/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for obtaining the HoG features for our car classifier training algorithm can be found in the `source` directory of the project, in the file `vehicle_features.py`. In this class, we create a class called `FeatureExtractor` that handles extracting both HoG features and color histogram features. This class is used to train our classifier in the Jupyter notebook called `Notebook.ipynb` that can also be found in the `source` directory of the project.

The `FeatureExtractor` is initialized with a training image, at which point HoG features are extracted using the `skimage.feature.hog` method.

The classifier is trained in `Notebook.ipynb` by collecting all the vehicle and non-vehicle images provided in links inside the project. Again, both HoG and color histogram features (in the [`YCrCb` color space](https://en.wikipedia.org/wiki/YCbCr)) are used to train our classifier, which was a [support vector machine (SVC)](https://en.wikipedia.org/wiki/Support_vector_machine) classifier.

During the training process, we use a `sklearn.preprocessing` `StandardScaler` to ensure that the dataset is standardized across all features (`StandardScaler` removes the mean, and sets the variance to unit variance).

We also use `train_test_split` to split the dataset into a training and testing dataset, in order to randomize our training samples and to check the accuracy of our classifier after training. Our classifier achieved 98.7% accuracy on the test set.

The following figure shows a sample of the training dataset (first car images, then non-car images), with HoG features and color histograms using the first color channel of the `YCrCb` color spaced version of the image.

For this figure, the HoG parameters are `orientations=10`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.

![A summary of features used to train our vehicle detection classifier][./figures/training-dataset-features.jpg]

We used the `YCrCb` color space as we anticipated vehicles would be best detected in a saturation-focused color channel (e.g. one of the color channels of `YCrCb`), as compared to a color space like `RGB` that has the saturation of the image split across all channels. All three channels of `YCrCb` were used in the final feature extraction.

The following figure shows the HoG visualization for entire test images from our driving video.

![HoG visualization for test images in the driving video][./figures/hog-images-00.jpg]

####2. Explain how you settled on your final choice of HOG parameters.

We started with HoG parameters suggested in the project premable on Udacity's website. Visually, the selected parameters seemed to most accurately capture the shape or identity of a vehicle without having too high of a resolution that would hinder performance. The final chosen parameters also seemed to do the best job when training our classifier.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

We trained a linear support vector machine (SVC) to identify cars vs non-cars with 98.7% accuracy. The process to train the SVM, including feature scaling, train/test split, and sample randomization are described in the above section.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

To find cars in a whole image, we use a sliding window technique to search relevant portions of the image for vehicles without creating an unreasonably large search space. 

First, we only search in the bottom half of the image, i.e., we don't search for cars above the horizon. Hardcoding this value would obviously cause issues if the viewport changed dramatically (e.g. going up or down hills).

Second, we scaled the window size according to the y-position of the window. For example, for sliding windows nearest the horizon, we chose a scale to make the sliding window smaller, as the effects of perspective will make potential candidate vehicles smaller in terms of how many pixels they represent in the image. For sliding windows closer to the vehicle (bottom of the image), cars appear larger, and the sliding window is appropriately larger. The scale of these boxes as a function of y-position was determined by trial and error, based on the sizes of the vehicles in the video.

Finally, choosing how much to shift the sliding window for each iteration was chosen by trial and error, so that vehicles were appropriately detected in video frames.

The figure below shows examples of our sliding window sizes at different y-positions and scales. Only a small sample of the total number of windows in the x-direction are showed, so that the sizes of the boxes can be best visualized.

![Sliding windows to isolate regions of the image to detect vehicles][./figures/windows.jpg]

Once sliding windows detected a vehicle, a heatmap method as described in the lectures was used to create regions of interest that might hold a vehicle. The union of detections was used to merge different boxes representing the same vehicle in a frame. Aberrant detections that appeared for a small number of frames were excluded.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![Example car detection heatmap][./figures/car-detections-00.jpg]
![Example car detection heatmap][./figures/car-detections-01.jpg]
![Example car detection heatmap][./figures/car-detections-02.jpg]
![Example car detection heatmap][./figures/car-detections-03.jpg]
![Example car detection heatmap][./figures/car-detections-04.jpg]
![Example car detection heatmap][./figures/car-detections-05.jpg]

As the reader can see, there are some false positives in these images. Since most of the false positives exist for a small number of frames, they are excluded from the image when processing multiple frames consecutively.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed collections of coordinates representing a bounding box to cover the area of each blob detected.

The value of the threshold depends on the number of already-processed frames. Once enough frames have been accumulated, the threshold is equal to ~10-20 frames; if a detection doesn't appear for this number of frames, it is excluded from the final display.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline still has a few spurious false positives, which could be improved. There are also a few short periods of time when cars are not detected.

This approach seems especially fragile if we deviate from the test video's driving circumstances: i.e., driving on a highway without any hills. For curvy roads, or driving at dramatically different speeds, assumptions about e.g. the position of the horizon and where vehicles may reside in the image may be broken and cause the pipeline to fail.

If our car detection training dataset excludes cars in orientations (e.g. facing the camera) or colors that are common when driving, our pipeline would fail to detect these.

There are also other classification and detection algorithms that may be more robust. There are deep convolutional neural network classification approaches that may be more robust than something like HoG. While HoG and color histograms are useful, these approaches require the programmer to hard-code features that they deem interesting. Alternatively, deep learning approaches discover the most relevant features during the training process, and may end up being more accurate. These approaches may also end up being more performant, which would be important for an algorithm requiring real-time processing of frames.
