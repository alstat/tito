import cv2, logging, time
import pandas as pd
from image_utils import *

logger = logging.getLogger(__name__)


def image_tag(image, face_paths, faces, face_prob, profile):
    '''
    '''
    #add patch to the image 
#    logger.info('Tagged {} faces from current image'.format(len(faces)))
    top = int(0 * image.shape[0])  # shape[0] = rows
    bottom = top
    left = 0
    right = int(0.40 * image.shape[1])
    borderType = cv2.BORDER_CONSTANT
    img = cv2.copyMakeBorder(image, top, bottom, left, right, borderType, None, (255,255,255))

    #font style
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.4

    count = 0 #height of profiles

    for items, face_path, prob in zip(faces, face_paths, face_prob):
        crop_param, name = items
        x, y, w, h = crop_param

        if prob < 0.15: break #name = 'Unknown'

        match_name = profile[profile['Name'].str.match(name.upper())]

        if count < 4:
            x_offset_face = image.shape[1] + 50
            y_offset_face = 180*count + 16

        elif count >= 4 and count < 8:
            x_offset_face = image.shape[1] + 250
            y_offset_face = 180*(count-4) + 16

        
        if not match_name.empty:
            #draw rectangle in the image
            cv2.rectangle(img,(x,y),(w+x,h+y),(0,255,0),2)

            #add resized detected face
            cropped_face = cv2.imread(face_path)
            resized_cropped_face = cv2.resize(cropped_face, (150, 150))
            img[y_offset_face:y_offset_face+resized_cropped_face.shape[0], x_offset_face:x_offset_face+resized_cropped_face.shape[1]] = resized_cropped_face

            #draw rectangle in the detected face
            cv2.rectangle(img,(x_offset_face,y_offset_face),(x_offset_face+resized_cropped_face.shape[1],y_offset_face+resized_cropped_face.shape[0]),(0,255,0),2)
            
            #font color of employee: green
            fontColor = (0, 255, 0)

            #write corresponding profile
#            text = str(match_name.as_matrix()[0][1]) + "\n" + str(match_name.as_matrix()[0][2]) + "\n" + str(match_name.as_matrix()[0][3]) + "\n" + str(match_name.as_matrix()[0][4])
            text = str(match_name.as_matrix()[0][1]) + "\n" + str(match_name.as_matrix()[0][2])
            y0, dy = 50, 15
            x_offset_profile = x_offset_face + 5
            y_offset_profile = y_offset_face + 65
            for i, line in enumerate(text.split('\n')):
                y = y0 + i*dy
                cv2.putText(img, str(line), (x_offset_profile,y_offset_profile+y+15), fontFace, fontScale, fontColor)
            

        else:
            #draw rectangle
            cv2.rectangle(img,(x,y),(w+x,h+y),(0,0,255),2)
            cv2.rectangle(img,(x,y),(w+x,h+y),(0,255,0),2)

            #add resized detected face
            cropped_face = cv2.imread(face_path)
            resized_cropped_face = cv2.resize(cropped_face, (150, 150))
            img[y_offset_face:y_offset_face+resized_cropped_face.shape[0], x_offset_face:x_offset_face+resized_cropped_face.shape[1]] = resized_cropped_face

            #draw rectangle in the detected face
            cv2.rectangle(img,(x_offset_face,y_offset_face),(x_offset_face+resized_cropped_face.shape[1],y_offset_face+resized_cropped_face.shape[0]),(0,0,255),2)
            cv2.rectangle(img,(x_offset_face,y_offset_face),(x_offset_face+resized_cropped_face.shape[1],y_offset_face+resized_cropped_face.shape[0]),(0,255,0),2)
            
            #font color of intruder: red
            fontColor = (0, 0, 255)
            fontColor = (0, 255, 0)

            #write corresponding profile
            text = str('ALERT!') + "\n" + str('Potential Intruder')
            text = str(name) + "\n" + str('Artist')
            y0, dy = 50, 15
            x_offset_profile = x_offset_face + 5
            y_offset_profile = y_offset_face + 65
            for i, line in enumerate(text.split('\n')):
                y = y0 + i*dy
                cv2.putText(img, str(line), (x_offset_profile,y_offset_profile+y+15), fontFace, fontScale, fontColor)
        count += 1

    return img

def tag_faces_UI(image_path,  face_paths, faces, face_prob, profiles):
    frame = cv2.imread(image_path)
    frame = image_tag(frame, face_paths, faces, face_prob, employee_profiles)
    output_file = os.path.join(args.output_dir, getFilename(image_path)+'_tagged.JPEG')
    cv2.imwrite(output_file,frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    # [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    # [cv2.IMWRITE_PNG_COMPRESSION, 9]

def build_video(input_directory):
    start_time = time.time()
    image_extensions = [".jpg", ".png", ".JPEG", ".PNG", ".JPG"]
    image_paths = getFilenameOfDatatype(input_directory, image_extensions)

#    logger.info('Read {} images from dataset'.format(len(image_paths)))
    
    frame = cv2.imread(image_paths[0])
    height, width, layers = frame.shape
    video_path = os.path.join('..\output', args.log_name, args.log_name+'_videoTagged.avi')
    #video_path = os.path.join(args.output_dir,args.log_name+'_videoTagged.avi')
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"XVID"), args.skip_rate,(width,height))

    basename = ''.join(getFilename(image_paths[0]).split('_')[:-2])
    splitPath, filename = os.path.split(image_paths[0])
#    ext = filename.split('_')[-1]

    for i, image_path in enumerate(image_paths):
#        print(os.path.join(splitPath, basename+'_'+str(i)+'_tagged.JPEG'))
        out.write(cv2.imread(os.path.join(splitPath, basename+'_'+str(i)+'_tagged.JPEG')))
    out.release()

#    video_filepath = os.path.join('..\output', args.log_name, args.log_name+'_videoTagged.avi')
    logger.info('Video saved in {}.'.format(video_path))
    logger.info('Video building: Completed in {} seconds'.format(time.time() - start_time))

def reformat_filename(filepath):
    splitPath, filename = os.path.split(filepath)
    filename = filename.split(".")

    return os.path.join(splitPath, filename)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--model-path', type=str, action='store', dest='model_path', help='Path to model protobuf graph')
    parser.add_argument('--input-dir', type=str, action='store', dest='input_dir', help='Input path of data to train on', default='../output/test')
    parser.add_argument('--employee-profiles', type=str, action='store', dest='employee_profiles', help='Input path of employee profiles', default='../dataset/profiles/NMDG_Profiles.csv')
    parser.add_argument('--video-results', type=str, action='store', dest='video_results', help='Input path of employee profiles', default='../output/results_video1.csv')
    parser.add_argument('--output-dir', type=str, action='store', dest='output_dir', default='../output/cctv_images')
    parser.add_argument('--log-name', type=str, dest='log_name', default='cctv')
    parser.add_argument('--skip-rate', type=int, dest='skip_rate', default=5)


    args = parser.parse_args()

    create_image_path(args.output_dir)

    employee_profiles = pd.read_csv(args.employee_profiles, names=['UID', 'Name', 'Position Title', 'Company', 'Gender'])
    image_faces = pd.read_csv(args.video_results)

    print(employee_profiles)
    #print(image_faces)
    
    image_extensions = [".jpg", ".png", ".JPEG", ".PNG", ".JPG"]
    image_paths = getFilenameOfDatatype(args.input_dir, image_extensions)

    logger.info('Read {} images from dataset'.format(len(image_paths)))

    start_time = time.time()
    for image_path in  image_paths:
        faces = []
        box = []
        image_path = os.path.join(args.input_dir, getFilename(image_path)+'.jpg')
        faces_all = image_faces.loc[image_faces['Filename'] == image_path].loc[:,'x':'h'].values
        tags_all = image_faces.loc[image_faces['Filename'] == image_path].loc[:,'Tag'].values
        boxes_all = image_faces.loc[image_faces['Filename'] == image_path].loc[:,'Cropped'].values
        probs_all = image_faces.loc[image_faces['Filename'] == image_path].loc[:,'Probability'].values

        for i in range(len(faces_all)):
            x,y,xx,yy = faces_all[i].tolist()
            bbox = [x, y, xx-x, yy-y]
            faces.append([bbox, ' '.join(tags_all[i].split('_'))])
            box.append(boxes_all[i])
        
        tag_faces_UI(image_path, box, faces, probs_all.tolist(), employee_profiles)

    logger.info('Frame tagging: Completed in {} seconds'.format(time.time() - start_time))

    build_video(args.output_dir)
