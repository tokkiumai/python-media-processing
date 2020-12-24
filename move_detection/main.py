import numpy as np

import argparse
import warnings
import datetime
import imutils
import json
import os
import time
import cv2

# construct the arg parser and parse them
ap = argparse.ArgumentParser()

ap.add_argument('-c', '--config', required=True,
                help='path to the JSON configuration file')

args = vars(ap.parse_args())

# filter warnings, load the config
warnings.filterwarnings('ignore')
config = json.load(open(args['config']))

# initialize the camera and grab a reference to the raw camera capture
# if the video arg is None, then we are reading from webcam
if not['use_ip_cam']:
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)
# otherwise, we are reading from a video input
else:
    camera = cv2.VideoCapture(config['ip_cam_addr'])

# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print('[INFO] warming up...')

time.sleep(config['camera_warmup_time'])
avg = None

lastUploaded = datetime.datetime.now()
motion_counter = 0
non_motion_timer = config['non_motion_timer']

fourcc = 0x00000020
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

writer = None
(h, w) = (None, None)

zeros = None
output = None
made_recording = False

# capture frames from the camera
while True:
    # grab the raw numpy array representing the image and initialize
    # the timestamp and occupied/unoccupied text
    (grabbed, frame) = camera.read()

    timestamp = datetime.datetime.now()
    motion_detected = False

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        print('[INFO] Frame couldn\'t be grabbed. Breaking - ' +
              datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
        break

    # resize the frame, convert it to grayscale and blur it
    frame = imutils.resize(frame, Width=confing['resize_width'])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the average frame is None, initialize it
    if avg is None:
        print('[INFO] starting background model...')
        avg = gray.copy().astype('float')
        continue

    # accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # treshold the delta image, dilate the threshold image to fill
    # in holes, then find contours on threshold image
    threshold = cv2.threshold(
        frameDelta, config['delta_threshold'], 255, cv2.THRESH_BINARY)[1]
    threshold = cv2.dilate(threshold, None, iterations=2)
    (cnts, _) = cv2.findContours(threshold.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the countors
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < conf['min_area']:
            continue

        # computethe bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w1, h1) = cv2.boundingRect(c)
        motion_detected = True

    fps = int(round(camera.get(cv2.CAP_PROP_FPS)))
    record_fps = 10
    ts = timestamp.strftime('%Y-%m-%d_%H_%M_%S')

    time_and_fps = ts + ' - fps: ' + str(fps)

    # draw the text and timestamp on the frame
    cv2.putText(frame, time_and_fps, (10,
                                      frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    def record_video():
        global writer, h2, w2, zeros, file_name, file_path

        if writer is None:
            filename = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
            file_path = (config['user_dir'] +
                         '{filename}.avi').format(filename=filename)

            (h2, w2) = frame.shape[:2]
            writer = cv2.VideoWriter(
                file_path, fourcc, record_fps, (w2, h2), True)
            zeros = np.zeros((h2, w2), dtype='uint8')

        # construct the final output frame, storing the original frame
        output = np.zeros((h2, w2, 3), dtype='uint8')
        output[0:h2, 0:w2] = frame

        # write the output frame to file
        writer.write(output)

    if motion_detected:
        motion_counter += 1

        # check to see if the number of frames wih motion is high enough
        if motion_counter >= config['mon_motion_frames']:
            if config['create_image']:
                image_path = (config['user_dir'] +
                              '/{filename}.jpg').format(filename=filename)
                cv2.imwrite(image_path, frame)

            record_video()

            made_recording = True
            non_motion_timer = config['non_motion_timer']

    # if there is no motion, continue recording until timer reaches 0
    # else clean everything up
    else:
        if made_recording is True and non_motion_timer > 0:
            non_motion_timer -= 1
            record_video()
        else:
            motion_counter = 0

            if writer is not None:
                writer.release()
                writer = None

            made_recording = False
            non_motion_timer = config['non_motion_timer']

    # check to see if the frames should be displayed to screen
    if config['show_video']:
        cv2.imshow('Security Feed', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

print('[INFO] cleaning up...')
camera.release()

cv2.destroyAllWindows()
