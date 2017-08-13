# -*- coding: utf-8 -*-
"""
Utilities Functions
camera caliration, color thresholding, grident, prespective tranfer, etc.
"""
import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt


def camera_cal(nx, ny, cam_pickle='camera_cal.p', testOff = True, displayOff = True):
  
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Make a list of calibration images
    images = glob.glob('camera_cal/cal*.jpg')
    img_test_fn = images[np.random.randint(0, len(images)-1)]
#    img_test_fn = images[0]
    img_test = cv2.imread(img_test_fn)
    img_size = (img_test.shape[1], img_test.shape[0])
    
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            if displayOff == False:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
                #write_name = 'corners_found'+str(idx)+'.jpg'
                #cv2.imwrite(write_name, img)
                cv2.imshow('img', img)
                cv2.waitKey(500)
            
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    
    if testOff == False:
        # save a test image
        dst = cv2.undistort(img_test, mtx, dist, None, mtx)
        fn = img_test_fn.split('\\')
        pn = fn[0]
        fn = fn[1].split('.')
        fn = pn + '\\' + fn[0] + '_undistored.' + fn[1]
        
        cv2.imwrite(fn,dst)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(img_test)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(dst)
        ax2.set_title('Undistorted Image', fontsize=30)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx # camera matrix
    dist_pickle["dist"] = dist # distroation coefficient
    pickle.dump( dist_pickle, open( cam_pickle, "wb" ) )
    return dist_pickle
        
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(dir_sobel)
    binary_output[(dir_sobel>= thresh[0]) & (dir_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output

def abs_sobel_thresh(img, orient='x', thresh =(0, 255)):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    if orient == 'xy':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

def color_thresh(chl, thresh = (150,255)):
    chl_binary = np.zeros_like(chl)
    chl_binary[(chl >= thresh[0]) & (chl <= thresh[1])] = 1
    return chl_binary

def warp(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size)
    return warped
    
################################################################
# image processing pipeline
def img_process(img, plotOn = False):
    # prespective transform
    src = np.float32(
                    [[210, 720],
                     [1250, 720],
                     [580, 460],
                     [760, 460]])
    dst = np.float32(
                    [[280, 720],
                     [1200, 720],
                     [280, 0],
                     [1200, 0]])
    
    warped = warp(img, src, dst)
    
    HLS = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS).astype(np.float)
    LAB = cv2.cvtColor(warped, cv2.COLOR_RGB2Lab).astype(np.float)

    HLS_L = HLS[:,:,1]
#    HLS_S = HLS[:,:,2]
    LAB_L = LAB[:,:,0]
    LAB_B = LAB[:,:,2]
    RGB_R = warped[:,:,0]

    # color thresholding
    HLS_L_binary = color_thresh(HLS_L, thresh = (190, 255)) # best white
#    HLS_S_binary = color_thresh(HLS_S, thresh = (120, 255)) 
    LAB_L_binary = color_thresh(LAB_L, thresh = (190, 255)) # best white
    LAB_B_binary = color_thresh(LAB_B, thresh = (140, 255)) # best yellow
    RGB_R_binary = color_thresh(RGB_R, thresh = (170, 255)) 
    
        
    # gridents
#    dir_grad = dir_threshold(img, sobel_kernel=15, thresh=(0.9, 1.1))
#    x_grad = abs_sobel_thresh(img, orient='x', thresh = (15, 200))
#    y_grad = abs_sobel_thresh(img, orient='y', thresh = (30, 255))
#    mag_grad = mag_thresh(img, sobel_kernel=3, mag_thresh=(90, 255))
#    y_grad = abs_sobel_thresh(img, orient='y', thresh = (50, 100))
   
    # combine channel
    comb1 = np.zeros_like(LAB_B_binary)
    comb2 = np.zeros_like(LAB_B_binary)
#    comb3 = np.zeros_like(LAB_B_binary)
    
    comb1[(LAB_L_binary == 1) | (LAB_B_binary == 1) | (HLS_L_binary == 1)] = 1 # best
    comb2[(LAB_B_binary == 1) | (RGB_R_binary == 1)] = 1       
    combined = comb1
             
    # stack channels
#    zero_chl = np.zeros_like(S)
#    color_binary1 = np.dstack((zero_chl, S_binary, x_grad))
    
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    unwarped_combined = warp(combined, dst, src)
    
    if plotOn == True:
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16, 6))
        f.tight_layout()
        
        ax1.imshow(HLS_L_binary)
        ax1.set_title('HLS_L_binary', fontsize=15)
        
        ax2.imshow(LAB_L_binary)
        ax2.set_title('LAB_L_binary', fontsize=15)
        
        ax3.imshow(LAB_B_binary)
        ax3.set_title('LAB_B_binary', fontsize=15)
        
        ax4.imshow(img)
        ax4.set_title('Original', fontsize=15)
        
        ax5.imshow(combined)
        ax5.set_title('warped_Combined', fontsize=15)
        
        ax6.imshow(unwarped_combined)
        ax6.set_title('unwarped_combined', fontsize=15)
        
        
        plt.subplots_adjust(left=0., right=0.6, top=0.9, bottom=0.)
    return combined, unwarped_combined, M, Minv
