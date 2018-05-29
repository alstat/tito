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

def preprocess(input_dir, output_dir, crop_dim, classifier, model):
    '''
    '''
    start_time = time.time()
    result_list = []
    start_time = time.time()
    pool = mp.Pool(processes=(mp.cpu_count()-1))

    create_output_directory(input_dir, output_dir)

    image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))

    for index, image_path in enumerate(image_paths):
        output_path = os.path.join(output_dir, os.path.basename(image_path)[:-4])

        result = pool.apply_async(preprocess_image, (image_path, output_path, crop_dim, classifier, model))

#        logger.info('Detected {} faces from image {}'.format(len(result.get()), os.path.basename(image_path)[:-4]))

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

def preprocess_image(image_path, output_path, crop_dim, classifier_filename, model_filename):
    '''
    Detect, align and crop faces and write output to output path.
    '''
    result = []
    faces = []
    image = None
    image = buffer_image(image_path)
    dets = detector(image, 1)

    if image is not None:
        if len(dets):
            for index, bbox in enumerate(dets):
                face = None
                x, y, l, r = bbox.left(), bbox.top(), bbox.right(), bbox.bottom()
                #face_bboxes.append([x, y, x-l, y-r])
                crop_img = image[y-40:r+40, x-40:l+40]

                face = align_image(crop_img, crop_dim)
                if face is None:
                    faces.append(face)
                    output_filename = save_face(output_path, image_path, face)
        #           label = get_label(get_embeddings(face,model_filename, sess), classifier_filename)

                    result.append([image_path, output_filename, x, y, l, r])
            logger.info('Detected {} faces from image {}'.format(len(dets), os.path.basename(image_path)[:-4])) 
        else:
            logger.info('Skipping filename {}'.format(image_path))
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

def identify_faces(input_directory, model_path, classifier_output_path, batch_size, num_threads, num_epochs,
         min_images_per_labels, split_ratio):
    '''
    Loads images from input_dir, creates embeddings using a model model_path, 
        and trains a classifier outputted to output_path.
    '''

    start_time = time.time()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        dataset = get_dataset(input_directory)
        #print(input_directory)
        images, labels, class_names, image_paths = load_data(dataset, image_size=160, batch_size=batch_size, num_threads=num_threads, num_epochs = 1)
        
        print(os.path.exists(model_path))
        load_model(model_filepath=model_path)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embedding_layer = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        filepath = []
        for image_path in image_paths:
            filepath.append(get_filename(image_path))

        emb_array, label_array = _create_embeddings(embedding_layer, images, labels, images_placeholder, phase_train_placeholder, sess)

        coord.request_stop()
        coord.join(threads=threads)

        logger.info('Read {} images from dataset'.format(len(emb_array)))

        load_model(model_filepath=model_path)
        logger.info('Loaded face recognition model')

        classifier_filename = classifier_output_path
        
        label_array = np.array(label_array.tolist(), dtype=int)
#        print(label_array, type(label_array))
#        print(image_paths, type(image_paths))

        results = _test_classifier(emb_array, label_array, classifier_filename, [ image_paths[i] for i in label_array])

        logger.info('Face detection: Completed in {} seconds'.format(time.time() - start_time))
    
    return results

def _create_embeddings(embedding_layer, images, labels, images_placeholder, phase_train_placeholder, sess):
    """
    Uses model to generate embeddings from :param images.
    :param embedding_layer: 
    :param images: 
    :param labels: 
    :param images_placeholder: 
    :param phase_train_placeholder: 
    :param sess: 
    :return: (tuple): image embeddings and labels
    """
    emb_array = None
    label_array = None
    try:
        i = 0
        while True:
            #images, labels = shuffle(images, labels, random_state=0)
            batch_images, batch_labels= sess.run([images, labels])
            logger.info('Processing iteration {} batch of size: {}'.format(i, len(batch_labels)))
            emb = sess.run(embedding_layer,
                           feed_dict={images_placeholder: batch_images, phase_train_placeholder: False})

            emb_array = np.concatenate([emb_array, emb]) if emb_array is not None else emb
            label_array = np.concatenate([label_array, batch_labels]) if label_array is not None else batch_labels
            i += 1

    except tf.errors.OutOfRangeError:
        pass

    return emb_array, label_array

def _test_classifier(emb_array, label_array, classifier_filename, image_paths):
    results = []
    logger.info('Evaluating classifier on {} images'.format(len(emb_array)))
    if not os.path.exists(classifier_filename):
        raise ValueError('Pickled classifier not found, have you trained first?')
    with open(classifier_filename, 'rb') as f:
        model, class_names = pickle.load(f)

        predictions = model.predict_proba(emb_array, )
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        for i in range(len(best_class_indices)):
            print('%s: \t\t %s: %.3f' % (image_paths[i], class_names[best_class_indices[i]], best_class_probabilities[i]))
            results.append([image_paths[i], class_names[best_class_indices[i]], str(best_class_probabilities[i])])


    return results


def load_model(model_filepath):
    '''
    Load frozen protobuf graph
    '''
    model_exp = os.path.expanduser(model_filepath)
    if os.path.isfile(model_exp):
        logging.info('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        logger.error('Missing model file. Exiting')
        sys.exit(-1)

def load_data(dataset, image_size, batch_size, num_threads, num_epochs, random_flip=False,
                            random_brightness=False, random_contrast=False):
    class_names = [cls.name for cls in dataset]
    image_paths, labels = lfw_input.get_image_paths_and_labels(dataset)
    labels = np.array(range(len(image_paths)))
#    print(image_paths)
    images, labels = lfw_input.read_data(image_paths, labels, image_size, batch_size, num_epochs, num_threads,
                                         shuffle=False, random_flip=random_flip, random_brightness=random_brightness,
                                         random_contrast=random_contrast)
    return images, labels, class_names, image_paths

def clean_directory(input_directory):
    image_paths  = glob.glob(os.path.join(input_directory, '**/*.jpg'))
    for image_path in image_paths:
        if os.path.getsize(image_path) < 100 * 1024:
            os.remove(image_path)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--model-path', type=str, action='store', dest='model_path',
                        help='Path to model protobuf graph')
    parser.add_argument('--input-dir', type=str, action='store', dest='input_dir',
                        help='Input path of data to train on')
    parser.add_argument('--output-dir', type=str, action='store', 
                        default='output', dest='output_dir')
    parser.add_argument('--batch-size', type=int, action='store', dest='batch_size',
                        help='Input path of data to train on', default=128)
    parser.add_argument('--num-threads', type=int, action='store', dest='num_threads', default=16,
                        help='Number of threads to utilize for queue')
    parser.add_argument('--num-epochs', type=int, action='store', dest='num_epochs', default=3,
                        help='Path to output trained classifier model')
    parser.add_argument('--split-ratio', type=float, action='store', dest='split_ratio', default=0.7,
                        help='Ratio to split train/test dataset')
    parser.add_argument('--min-num-images-per-class', type=int, action='store', default=1,
                        dest='min_images_per_class', help='Minimum number of images per class')
    parser.add_argument('--classifier-path', type=str, action='store', dest='classifier_path',
                        help='Path to output trained classifier model', default='../output/')
    parser.add_argument('--crop-dim', type=int, action='store', default=180, dest='crop_dim',
                        help='Size to crop images to')
    parser.add_argument('--employee-profiles', type=str, action='store', dest='employee_profiles', 
                        help='Input path of employee profiles', default='../data/profiles/NMDG_Profiles.csv')
    parser.add_argument('--log-dir', type=str, action='store', dest='log_dir',default='nmdg',
                        help='Size to crop images to')

    args = parser.parse_args()

    image_tags = identify_faces(input_directory=args.output_dir, 
                            model_path=args.model_path, 
                            classifier_output_path=args.classifier_path,
                            batch_size=args.batch_size, 
                            num_threads=args.num_threads, 
                            num_epochs=args.num_epochs,
                            min_images_per_labels=args.min_images_per_class, 
                            split_ratio=args.split_ratio)
    
    image_tags_df = pd.DataFrame(np.array(image_tags), columns=['Cropped', 'Tag', 'Probability'])
    image_tags_df.to_csv('../output/'+args.log_dir+'/results_video_tags_'+args.log_dir+'.csv', index=False)

    image_faces_df = pd.read_csv('../output/'+args.log_dir+'/results_video_faces_'+args.log_dir+'.csv')
    image_faces_df = image_faces_df.merge(image_tags_df, left_on='Cropped', right_on='Cropped')
    image_faces_df.to_csv('../output/'+args.log_dir+'/results_video_'+args.log_dir+'.csv', index=False)


    image_tags_df = pd.DataFrame(np.array(image_tags), columns=['Cropped', 'Tag', 'Probability'])

    logging.info('Total number of faces detected:  {} '.format(len(image_tags_df)))
    
    # Save the results
#    image_tags_df.to_csv('../output/'+args.log_dir+'/results_video_tags'+args.log_dir+'.csv', index=False)
