## Filename: image_utils.py
## Author: Maria Isabel S. Salydares

import numpy as np
import argparse, csv
import cv2, os
import timeit
import pandas as pd

def getFilenameOfDatatype(inputDir, fileExtensions):
	'''
	Gets the filenames given a set of file extensions.
	'''
	import os
	filenames = []
	#inputDir = "./input"

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

def getFilename(filepath):
	'''
	Get the filename without the extension
	'''
	splitPath, filename = os.path.split(filepath)
	filename = filename.split(".")

	return filename[0]

def write2CSV(dataset, filename):
	'''
	Write to a CSV file
	'''
	myFile = open(filename, 'wb')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(dataset)

def loadImage(img):
	if cv2.imread(img) is not None:
		image = cv2.imread(img)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		return image, gray
	else:
		return [], []

def findLatestObjectCount(imageFolder, prefix):
	files = [filename for filename in os.listdir(imageFolder) if filename.startswith(prefix)]
	maxCount = 0
	
	for fname in files:
		if fname is None:
			maxCount = 0
		else:
			fname = fname.split("_")
			fname = fname[-1].split(".")
			if int(fname[0]) > maxCount:
				maxCount = int(fname[0])

	return maxCount+1

def create_image_path(imageDirectory):
	'''
	'''
	if not os.path.exists(imageDirectory): os.makedirs(imageDirectory)

def joinBoxes(a, b):
	if len(a)!=0 :
		if len(b)!=0 :
			return np.concatenate((a,b), axis=0)
		else:
			return a
	elif len(b)!=0 : #is not None
		return b
	else:
		"Empty lists."
		return []

def displayBoundary(image, bboxes):
	for (x,y,w,h) in bboxes:
		cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 3)

	print("Bounding boxes displayed. Click 'q' to exit.")
	cv2.imshow("Detected and selected faces and objects", image)
	while True:
		# display the image and wait for a keypress
		key = cv2.waitKey(1) & 0xFF

		# if the 'q' key is pressed, break from the loop
		if key == ord("q"):
			break

	return image

def formatCroppingData(path, bboxes, labels, lastCount):
	dataset = []
	counts = range(lastCount-len(bboxes), lastCount+1)
	for (bbox, label, count) in zip(bboxes, labels, counts):
		x,y,w,h = bbox
		cropped = label+"_"+str(count)+".png"
		dataset.append([path, cropped, label, x , y , w, h])

	return dataset

def crop_images_per_tag(tag):
	'''
	'''
	cwd = os.getcwd()	
	imgDir = cwd+"\\images\\"+artist
	vidDir = cwd+"\\videos\\"+artist

	videos = get_video_list(vidDir)
	create_image_path(imgDir)
	videos_to_images(videos, 3, imgDir)

def detectFaces(image, faceCascade):
	'''
	Implements face detection based from the xml file from opencv
	'''

	faces = faceCascade.detectMultiScale(
		image,
		scaleFactor=1.1,
		minNeighbors=7,
		minSize=(30, 30),
		flags = cv2.CASCADE_SCALE_IMAGE
    )

	#print("Found {0} faces!".format(len(faces)))
	#print faces

	return faces

def selectFaces(imagePath, faceCascade):
	faces = []
	image , gray = loadImage(imagePath)

	#print("\nDetecting faces...")
	faces = detectFaces(gray, faceCascade)
	#taggedFaces = displayBoundary(image, faces)

	return faces

def saveTaggedImage(imagePath, outputDir, objects, tags):
	suffix = 0
	image , gray = loadImage(imagePath)

	for ((x,y,w,h), tag) in zip(np.array(objects, dtype=int), tags):
		crop_img = image[y:y+h, x:x+w]
		suffix = findLatestObjectCount(outputDir, tag)
		filename = os.path.join(outputDir,tag+"_"+str(suffix)+".png")
		cv2.imwrite(filename, crop_img)

	return suffix