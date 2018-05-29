import cv2, logging, time
import pandas as pd
from image_utils import *
from datetime import datetime


logger = logging.getLogger(__name__)

def add_time_result(date_time, results_video_face, time_interval):
    t_i, d, time_sec = date_time
    #print(datetime.fromtimestamp(time_sec).strftime("%A, %B %d, %Y %I:%M:%S:%f"))
    #new column
    year = ['Year'] 
    month = ['Month']
    day = ['Day']
    hour = ['Hour']
    minute = ['Minute']
    second = ['Second']
    microsecond = ['Microsecond']

    #back-calculate the time from filename
    filename = results_video_face['Filename']
    sep1 = '\\'
    sep2 = '.jpg'
    sep3 = '_'

    for i in range(1,len(filename)): #get seconds per frame
        rest = filename[i].split(sep1,1)[1]
        rest = rest.split(sep2,1)[0]
        rest = float(rest.split(sep3,1)[1])*time_interval + time_sec
        #print(rest)
        year.append(str(datetime.fromtimestamp(rest).year))
        month.append(str(datetime.fromtimestamp(rest).month))
        day.append(str(datetime.fromtimestamp(rest).day))
        hour.append(str(datetime.fromtimestamp(rest).hour))
        minute.append(str(datetime.fromtimestamp(rest).minute))
        second.append(str(datetime.fromtimestamp(rest).second))
        microsecond.append(str(datetime.fromtimestamp(rest).microsecond))

    #add new column for time with random numbers
    results_video_face['year'] = year
    results_video_face['month'] = month
    results_video_face['day'] = day
    results_video_face['hour'] = hour
    results_video_face['minute'] = minute
    results_video_face['second'] = second
    results_video_face['milli'] = microsecond

    results_video_face = results_video_face.iloc[1:]

    return results_video_face

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--employee-profiles', type=str, action='store', dest='employee_profiles', help='Input path of employee profiles', default='../data/profiles/NMDG_Profiles.csv')
    parser.add_argument('--video-results', type=str, action='store', dest='video_results', help='Input path of employee profiles', default='../output/results_video1.csv')

    args = parser.parse_args()

    employee_profiles = pd.read_csv(args.employee_profiles, names=['UID', 'Name', 'Position Title', 'Company', 'Gender'])

    #csv of face results
    results_names = ['Filename', 'Cropped', 'x', 'y', 'w', 'h', 'Tag', 'Probability']
    results_faces = pd.read_csv(args.video_results, names=results_names)

    #start time of face detection
    tt_i = datetime.now().strftime('%Y/%m/%d %H:%M:%S:%f')
    dd = datetime.strptime(tt_i, "%Y/%m/%d %H:%M:%S:%f")
    ttime_sec = time.mktime(dd.timetuple())

    tito = add_time_result([tt_i, dd, ttime_sec,], results_faces, 1./10.)
    tito.to_csv(args.video_results[:-4]+'_tito'+ '.csv')

    logger.info('Time-in, time-out record save in {}.'.format(args.video_results[:-4]+'_tito'+ '.csv'))