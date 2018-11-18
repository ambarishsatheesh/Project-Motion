import cv2
import numpy as np

# Create a VideoCapture object and read from input file, store in 'cap' variable
# If the input is taken from the camera, pass 0 instead of the video file name.
cap = cv2.VideoCapture("D:/Videos/cars.mp4")

# Background subtraction function used to create background object. This code implements GMM (Gaussian Mixture Models). See OpenCV documentation for detailed info.
# MOG2 algorithm is used here. Parameter 'history' determines how many previous frames are used to build background model.
# If an object in video is still for that amount of frames, it will disappear into the background.
# 'varThreshold' is the threshold on the squared Mahalanobis distance between the pixel and the model. Decides whether a pixel is well described by the background model.
# 'detectshadows' detects and marks shadows but decreases speed.
#subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=200, detectShadows=False)

# Duplicate subtraction function to compare noise reduction
subtractor2 = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=200, detectShadows=False)

# A kernel is the structuring element applied to the input image by the morphological operation to generate the output. Here an ellipse-shaped kernel is used.
# Cross, circular or rectangular shapes can also be used. a 3x3 matrix is used here since it seems to provide optimal noise reduction.
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Different subtraction algorithm: KNN.
# 'dist2Threshold' is the threshold on the squared distance between the pixel and the sample to decide whether a pixel is close to that sample.
subtractor3 = cv2.createBackgroundSubtractorKNN(history=20, dist2Threshold=2000, detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

while True:                 # while VideoCapture object is being read

    # 'cap.read()' returns a bool and frame. The next frame is captured from the 'cap' variable.
    # Returns a tuple which is unpacked into the two variables '_' (bool), and 'frame'. '_' is just to signify that the bool is being ignored.
    # Using this line multiple times causes skipping of frames, speeding up the video.
    _, frame = cap.read()

    #mask = subtractor.apply(frame)

    fmask = subtractor2.apply(frame)  # .apply() applies the background subtraction function to the frame

    # 'morphologyEX()' is a function used for the morphological operation 'Opening' which is erosion followed by dilation. Helps remove noise.
    # This function can also be used with 'MORPH_CLOSE' for closing operations (dilation followed by erosion). Helps in closing up small holes within foreground.
    # 'MORPH_TOPHAT', 'MORPH_BLACKHAT' and 'MORPH_GRADIENT' can also be used.
    # Better to apply operation to the result of the subtraction to remove noise.
    fmask = cv2.morphologyEx(fmask, cv2.MORPH_OPEN, kernel)

    fmask2 = subtractor3.apply(frame)                           # Applies the background subtraction function to the frame
    fmask2 = cv2.morphologyEx(fmask2, cv2.MORPH_OPEN, kernel)   # Executes morphological operation to the result of the subtraction

    combined = np.hstack((fmask, fmask2))                       # Stacks video frames horizontally

    cv2.imshow("KNN+MOG2Morph", combined)                       # display frames in window with title
    #cv2.imshow("MOG2", mask)
    #cv2.imshow("mask", mask)


    # 'waitkey()' function waits for a key event for a delay of milliseconds (or infinity when value is 0).
    # This expression calculates the milliseconds required between frames to display at normal speed based on the frame rate of the source video.
    # In reality this will run slower than the original speed due to processing time. Use QT or lower-level APU for GUI implementation.
    key = cv2.waitKey(int((1/int(60))*1000)) & 0xFF

    # If escape key is pressed, break.
    if key == 27:
        break

cap.release()               # Releases capture
cv2.destroyAllWindows()
