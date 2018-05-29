import cv2, logging, time
import pandas as pd
from image_utils import *

logger = logging.getLogger(__name__)

def build_video(input_directory, output_directory, log_name):
    start_time = time.time()
    image_extensions = [".jpg", ".png", ".JPEG", ".PNG", ".JPG"]
    image_paths = getFilenameOfDatatype(input_directory, image_extensions)

#    logger.info('Read {} images from dataset'.format(len(image_paths)))
    
    frame = cv2.imread(image_paths[0])
    height, width, layers = frame.shape
    video_path = os.path.join(output_directory,log_name+'.mp4')
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"XVID"), 5,(width,height))

    basename = ''.join(getFilename(image_paths[0]).split('_')[:-1])
    splitPath, filename = os.path.split(image_paths[0])

    for i, image_path in enumerate(image_paths):
#        print(os.path.join(splitPath, basename+'_'+str(i)+'.jpg'))
        out.write(cv2.imread(os.path.join(splitPath, basename+'_'+str(i)+args.image_type)))
    out.release()

    logger.info('Video saved in {}.'.format(os.path.join(splitPath, basename+'_'+str(i)+'.png')))
    logger.info('Video building: Completed in {} seconds'.format(time.time() - start_time))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input-dir', type=str, action='store', dest='input_dir', help='Input path of data to train on', default='../output/test')
    parser.add_argument('--output-dir', type=str, action='store', dest='output_dir', default='../output/cctv_images')
    parser.add_argument('--log-name', type=str, action='store', dest='log_name', default='nmdg')
    parser.add_argument('--image-type', type=str, action='store', dest='image_type', default='.png')

    args = parser.parse_args()

    create_image_path(args.output_dir)

    build_video(args.input_dir, args.output_dir, args.log_name)