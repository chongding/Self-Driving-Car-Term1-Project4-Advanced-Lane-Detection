# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 20:30:15 2017

@author: chong
"""
from utilites import camera_cal, img_process
from line_detection import linedet_win_sliding, linedet_conv, lane_polyfill, curv_cal
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os
from Line import Line


def video_process(image):
    cam_cal = False     # camera calibration
    if cam_cal == True:
        dist_pickle = camera_cal(9,6, testOff = True)
    else:
        dist_pickle = pickle.load( open( "camera_cal.p", "rb" ) )
    
    mtx = dist_pickle["mtx"]  # camera matrix
    dist = dist_pickle["dist"]  # distroation coefficient
    
#    lines = Line(num_frame) # line class to store historical line_fits
                  
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    warped_combined, unwarped_combined, M, Minv = img_process(dst)
        
    try:
        left_fit, right_fit, lane_cur, lane_area = linedet_win_sliding(warped_combined)
        
        if lines.store_counter == 0:
            lines.store(left_fit, right_fit, lane_cur, lane_area) # line_fits stored
            lines.last(left_fit, right_fit, lane_cur, lane_area)
        else:
            if lines.check(lane_cur, lane_area, 0.3): # if too much change, prorably bad detection
                lines.store(left_fit, right_fit, lane_cur, lane_area) # line_fits stored
                lines.last(left_fit, right_fit, lane_cur, lane_area)
            else:
#                print(lines.area_avg)
#                print(lane_area)
                lines.last(lines.l_fit_avg, lines.r_fit_avg, lines.cur_avg, lines.area_avg)
  
        dst_filled = lane_polyfill(dst, lines.l_fit_last, lines.r_fit_last, lines.last_cur, Minv)
                    
    except:
        dst_filled = lane_polyfill(dst, lines.l_fit_last, lines.r_fit_last, lines.last_cur, Minv)

#        f, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(16, 7))
#        f.tight_layout()
#
#        ax1.imshow(dst)
#        ax1.set_title('problem', fontsize=20)
#        
#        ax2.imshow(warped_combined)
#        ax2.set_title('warped_combined', fontsize=20)
#        f.savefig('error.jpg')
#    
    return dst_filled  

########################################################                                                       
# video processing parameters
num_frame = 15 # number of frames to store
forget_coef = 1 # use to weight the history and latest lines

lines = Line(num_frame)
dir_output = 'output_videos/'
video_name = 'challenge_video.mp4'
video_output = os.path.join(dir_output, video_name)

clip1 = VideoFileClip(os.path.join("test_videos/", video_name))#.subclip(38,49)
white_clip = clip1.fl_image(video_process) #NOTE: this function expects color images!!
white_clip.write_videofile(video_output, audio=False)