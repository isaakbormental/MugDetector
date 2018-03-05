**MUG Detector**
The script1.py contains a training + detection procedures, works with vod.ogv video.

**Methods**
The *main* method :
firstly calls load_train_images() method, then converts images to HOG features, transforms to vector, creates labels. Then the LinearSVC classifier and trains it. Funally the video capture opens the video and to each frame it applies find_mugs() method, wich also draws rectangles. The resize is done (along with convertion to grayscale) to make computations less expensive.

*load_train_images()* : 
gathers paths to all images paths from mug and no_mug folders and reads images with imread (grayscale). Returns two arrays with gray images: mug_images, non_mug_images

*images2HOGfeatures(mugs_array, non_mug_array)* :
iterates through all images in two arrays and converts each to HOG. returns two hog arrays: mug_features, non_mug_features

*resize_frame(frame, ratio)* :
Applied to frames from video to decrease computational cost. returns resized frame

*find_mugg(image, classif)* :
contains sliding windows approach, to each window makes classifier prediction. returns image with rectangles.

*save_frames(x_s, y_s, window)* :
used to extract patches from video frames to gather dataset for training