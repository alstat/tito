#videoParser.py
## Author: Maria Isabel Saludares
'''
Video parser software

This reads the video file/files (in a directory) and saves the frames into images.
The saving of images has one parameter: number of frames to skip
for saving.  The saved images are immediately saved in the in the
same folder name as its original video file but inside the output folder

Usage:
	video_parser.py [-v] [-i] [-k] [-s]

'''

import cv2
from skimage import data, io, filters
import matplotlib.pyplot as plt
import numpy as np
import os, argparse, logging, time
from image_utils import *

logger = logging.getLogger(__name__)

def get_video_list(videoDirectory):
	'''
	'''
	videos = []
	if os.path.exists(videoDirectory):
		for root, dirs, files in os.walk(videoDirectory):
			for file in files:
				if file.endswith(".mp4") | file.endswith(".MOV"):
					logger.info('Video found in folder: {} '.format(file))
					videos.append(os.path.join(root, file))
	else:
		logger.info("No videos found in the directory.")
	return videos

def create_image_path(imageDirectory):
	'''
	'''
	if not os.path.exists(imageDirectory): os.makedirs(imageDirectory)

def rotate_image(image, angle):
	(h, w) = image.shape[:2]
	center = (w / 2, h / 2)
	 
	# rotate the image by 180 degrees
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated = cv2.warpAffine(image, M, (w, h))

	return rotated

def videos_to_images(videos, skipFrame, imagePath):
	'''
	'''
	start_time = time.time()
	count = 0
	imgCount = 0

	for video in videos:
		cap = cv2.VideoCapture(video)
		success, image = cap.read()
		
		logger.info("Parsing video {} . . .".format(getFilename(video)))
		
		while success:
			success, image = cap.read()
			if image is None:
				break
			#image = rotate_image(image, 180)
			if count % args.skip == 0:
				cv2.imwrite(os.path.join(imagePath,(artist+"_%d.jpg" % imgCount)), image)
				imgCount += 1

			if cv2.waitKey(10) == 27: 	# exit if Escape is hit
				break

			count += 1
	logger.info("{} Images saved in {}".format(imgCount-1, imagePath))
	logger.info('Completed in {} seconds'.format(time.time() - start_time))

def parse_videos_per_artist(artist):
	'''
	'''
	cwd = os.getcwd()	
	imgDir = args.image_dir
	vidDir = args.videos_dir

	videos = get_video_list(vidDir)
	create_image_path(imgDir)
	videos_to_images(videos, 3, imgDir)

if __name__ == "__main__":
	print(__doc__)
	logging.basicConfig(level=logging.INFO)

	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--videos', dest='videos_dir', default='../dataset/videos/', help='Path to videos')
	parser.add_argument('-i', '--images', dest='image_dir', default='../output/EXO', help='Path to save images')
	parser.add_argument('-k', '--keywords', help='delimited list input', type=str, required=True)
	parser.add_argument('-s', '--skip', default=3, help='Skip count for saving frames', type=int, required=True)
	args = parser.parse_args()

	artists = [str(item) for item in args.keywords.split(',')]
	skipCount = args.skip

	for artist in artists:
		parse_videos_per_artist(artist)

# os.path.normpath(path).lstrip(os.path.sep).split(os.path.sep)
# os.path.normpath(a_path).split(os.path.sep)