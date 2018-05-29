## Filename: webcamDisplay.py
## Author: Maria Isabel Saludares
## Brief: CS282 Programming Assignment 1

'''
Video capture sample.

Reads input from webcam and displays:
	image: 		frames captured from WebCamDisplay
	grayscale: 	grayscale of this frame
	canny: 		Canny edge detection on frame

These three (3) images are displayed side-by-side
(image-grayscale-canny) in a single window.

Usage:
    webcamDisplay.py

Keys:
    q    - exit

'''

import numpy as np
import cv2
import sys
import os
import dlib
import glob
from skimage import io

predictor_path = "../etc/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
#win = dlib.image_window()

def nothing(x):
	pass

def grayTo3D(grayImg):
	'''
	Converts a gray image (1D) to a color image (3D) by using
	the same gray image in the three channels
	'''
	colorImg = np.ones((grayImg.shape[0], grayImg.shape[1], 3))
	colorImg[:,:,0] = grayImg
	colorImg[:,:,1] = grayImg
	colorImg[:,:,2] = grayImg

	return colorImg

def GetFilenameOfDatatype(fileExtensions):
	'''
	Gets the filenames given a set of file extensions.
	'''
	import os
	filenames = []
	inputDir = "./input"

	numOfFiles = 0
	for ext in fileExtensions:
		files = [f for f in os.listdir(inputDir) if f.endswith(ext)]
		if len(files) != 0:
			numOfFiles = numOfFiles + len(files)
			for filename in files:
				filenames.append(inputDir+"/"+filename) 
	
	filenames = np.array(filenames)
	filenames.reshape(numOfFiles,1)

	return filenames

def WebCamDisplay():
	'''
	Reads input from webcam and displays:
		(1) 	frames captured from WebCamDisplay
		(2)		grayscale of this frame
		(3)		Canny edge detection on frame

	Default min,max values for Canny edge detection is 50,150.
	'''
	cap= cv2.VideoCapture(0)
	ret, frame = cap.read()

	frameShape = frame.shape
	finalWindow = np.zeros((frameShape[0], frameShape[1]*3, frameShape[2]))

	minGrayThresh = 50
	maxGrayThresh = 150

	while(True):
	    # Capture frame-by-frame
	    ret, frame = cap.read()

	    # grayscal and canny edge detection per frame
	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
	    cannyEdges = cv2.Canny(gray, minGrayThresh, maxGrayThresh)

	    gray = grayTo3D(gray)
	    cannyEdges = grayTo3D(cannyEdges)

	    imageList = [frame/255., gray/255., cannyEdges/255.]
	    for i in range(3):
	    	a = (i)*frame.shape[1]
	    	b = (i+1)*frame.shape[1]
	    	finalWindow[:,a:b,:] = imageList[i] # Rescale [0 255] -> [0 1]

	    cv2.imshow('Display', finalWindow)
	    #cv2.imshow('Display', detect_faces(frame))

	    if cv2.waitKey(1) == ord("q"):
	        break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

def detect_faces(img):
	#win.clear_overlay()
	#win.set_image(img)
	bboxes = []

	# Ask the detector to find the bounding boxes of each face. The 1 in the
	# second argument indicates that we should upsample the image 1 time. This
	# will make everything bigger and allow us to detect more faces.
	dets = detector(img, 1)
	#print("Number of faces detected: {}".format(len(dets)))
	for k, d in enumerate(dets):
		#print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
		# Get the landmarks/parts for the face in box d.
		shape = predictor(img, d)
		#print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
		cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 255), 2)
		bboxes.append([d.left(), d.top(), d.right(), d.bottom()])
		# Draw the face landmarks on the screen.
		#win.add_overlay(shape)

	#win.add_overlay(dets)
	#dlib.hit_enter_to_continue()

	return img



if __name__ == '__main__':
    import sys
    import getopt

    print(__doc__)

    WebCamDisplay()