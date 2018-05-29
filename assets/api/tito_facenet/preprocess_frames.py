import argparse
import logging, time
import glob, os, pickle, sys
import cv2, dlib, csv, random
import numpy as np
import multiprocessing as mp
import pandas as pd

import tensorflow as tf
from sklearn.svm import SVC
from tensorflow.python.platform import gfile

from lfw_input import filter_dataset, split_dataset, get_dataset
import lfw_input
from align_dlib import AlignDlib


logger = logging.getLogger(__name__)

align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), '../etc/shape_predictor_68_face_landmarks.dat'))
predictor_path = "../etc/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def create_output_directory(input_dir, output_dir):
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    for image_dir in os.listdir(input_dir):
        image_name = os.path.basename(os.path.basename(image_dir))[:-4]
        image_output_dir = os.path.join(output_dir, image_name)

        if not os.path.exists(image_output_dir): 
            os.makedirs(image_output_dir)

    logger.info('Read {} images from dataset'.format(len(os.listdir(input_dir))))

def preprocess(input_dir, output_dir, crop_dim):
    '''
    '''
    result_list = []
    start_time = time.time()
    pool = mp.Pool(processes=(mp.cpu_count()))

    create_output_directory(input_dir, output_dir)

    image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))

    start_time = time.time()
    for index, image_path in enumerate(image_paths):
        output_path = os.path.join(output_dir, os.path.basename(image_path)[:-4])
        result = pool.apply_async(preprocess_image, (image_path, output_path, crop_dim))
        result_list.append(result.get())

    pool.close()
    pool.join()

    logger.info('Pre-processed {} images from dataset'.format(len(image_paths)))
    logger.info('Pre-processing: Completed in {} seconds'.format(time.time() - start_time))

    return fix_array(result_list)

def buffer_image(filename):
    logger.debug('Reading image: {}'.format(filename))
    image = cv2.imread(filename, )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def preprocess_image(image_path, output_path, crop_dim):
    '''
    Detect, align and crop faces and write output to output path.
    '''
    result = []
    faces = []
    image = None
    image = buffer_image(image_path)

    if image is not None:
        dets = detector(image, 1) 
        for index, bbox in enumerate(dets):
            face = None
            x, y, l, r = bbox.left(), bbox.top(), bbox.right(), bbox.bottom()
            #face_bboxes.append([x, y, x-l, y-r])
            crop_img = image[y-40:r+40, x-40:l+40]

            face = align_image(crop_img, crop_dim)
            if face is not None:
                faces.append(face)
                output_filename = save_face(output_path, image_path, face)
                result.append([image_path, output_filename, x, y, l, r])
        logger.info('Detected {} faces from image {}'.format(len(dets), os.path.basename(image_path)[:-4])) 

    else:
        logger.info('Skipping filename {}'.format(image_path))

    return result

def align_image(image, crop_dim):
    bb = align_dlib.getLargestFaceBoundingBox(image)
    aligned = align_dlib.align(crop_dim, image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    if aligned is not None:
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return aligned

def save_face(output_dir, filename, image):
    suffix = find_lates_count(output_dir, get_filename(filename)) #random.randint(1,1000) #i #
    filename = os.path.join(output_dir,get_filename(filename)+"_"+str(suffix)+".jpg")
    cv2.imwrite(filename, image)

    return filename

def get_filename(filepath):
    splitPath, filename = os.path.split(filepath)
    filename = filename.split(".")

    return filename[0]

def fix_array(dataset):
    results = []
    for perFrame in dataset:
        for perFace in perFrame:
            results.append(perFace)

    return results

def find_lates_count(imageFolder, prefix):
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

def clean_directory(input_directory):
    image_paths  = glob.glob(os.path.join(input_directory, '**/*.jpg'))
    for image_path in image_paths:
        if os.path.getsize(image_path) < 100 * 1024:
            os.remove(image_path)

def create_path(directory):
    if not os.path.exists(directory): os.makedirs(directory)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input-dir', type=str, action='store', dest='input_dir',
                        help='Input path of data to train on')
    parser.add_argument('--output-dir', type=str, action='store', 
                        default='output', dest='output_dir')
    parser.add_argument('--crop-dim', type=int, action='store', default=180, dest='crop_dim',
                        help='Size to crop images to')
    parser.add_argument('--log-dir', type=str, action='store', dest='log_dir',default='nmdg',
                        help='Size to crop images to')
    args = parser.parse_args()

    create_path('../output/'+args.log_dir)

    image_faces = preprocess(args.input_dir, args.output_dir, args.crop_dim)
    
    # Save results
    image_faces_df = pd.DataFrame(np.array(image_faces), columns=['Filename', 'Cropped', 'x', 'y', 'w', 'h'])
    image_faces_df.to_csv('../output/'+args.log_dir+'/results_video_faces_'+args.log_dir+'.csv', index=False)

    logging.info('Total number of faces detected:  {} '.format(len(image_faces_df)))
