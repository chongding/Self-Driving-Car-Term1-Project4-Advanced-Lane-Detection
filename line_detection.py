'''
Find lines
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

def linedet_win_sliding(binary_warped, nwindows = 20, plotOn = False):
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 40
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    lane_cur = curv_cal(left_fit, right_fit, 720)
    lane_area = area_cal(left_fit, right_fit, 720)
    
    if plotOn == True:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='red')
        plt.plot(right_fitx, ploty, color='green')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)        
    return left_fit, right_fit, lane_cur, lane_area

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output


def linedet_conv(warped, window_width = 50, window_height = 80, margin = 100, plotOn = False):
    warpage = np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
#        find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
	    # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
	    # Add what we found for that layer
        window_centroids.append((l_center,r_center))
        leftx = []
        lefty = []
        rightx = []
        righty = []

    if len(window_centroids) > 0:   
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)
    
        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
    	     # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
            leftx.append(window_centroids[level][0])
            lefty.append(level*window_height + window_height/2)
            rightx.append(window_centroids[level][1])
            righty.append(level*window_height + window_height/2)
    
        # Draw the results
        template_l = np.array(l_points,np.uint8) # add both left and right window pixels together
        template_r = np.array(r_points,np.uint8) # add both left and right window pixels together
   
        warpage[template_l == 255] = [255, 0, 0]
        warpage[template_r == 255] = [0, 0, 255]
        
        left_fit = np.polyfit(lefty[::-1], leftx, 2)
        right_fit = np.polyfit(righty[::-1], rightx, 2)
     
    
    if plotOn == True:
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        plt.imshow(warpage)
        plt.plot(left_fitx, ploty, color='red')
        plt.plot(right_fitx, ploty, color='green')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)   
        plt.title('window fitting results')
        plt.show()
    
    return left_fit, right_fit

def curv_cal(left_fit, right_fit, ymax=720):
    ploty = np.linspace(0, ymax-1, num=ymax)
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radious of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*ymax*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*ymax*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    lane_cur = np.mean([left_curverad, right_curverad])
    
    return lane_cur
    
def area_cal(left_fit, right_fit, ymax=720):
    ploty = np.linspace(0, ymax-1, num=ymax)/ymax
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]    
    area = np.dot((rightx - leftx), ploty)
    return area
    
def off_center(leftx, rightx, xmax, xm_per_pix = 3.7/700):
    offset = (xmax/2-np.mean([leftx[-1], rightx[-1]]))*xm_per_pix
    if offset > 0:
        offset = 'Vehicle is ' + str(round(abs(offset), 2)) + 'm right to the center'
    else:
        offset = 'Vehicle is ' + str(round(abs(offset), 2)) + 'm left to the center'
    return offset
    
def lane_polyfill(dst, left_fit, right_fit, lane_cur, Minv, ymax=720):
    filled = np.zeros_like(dst)
    ploty = np.linspace(0, ymax-1, num=ymax)
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    left_pts = np.transpose(np.array([leftx, ploty]))
    right_pts = np.transpose(np.array([rightx, ploty]))
    pts = np.array(np.vstack((left_pts, np.flipud(right_pts))), dtype = int)
    cv2.polylines(filled, np.int_([pts]), isClosed=False, color=(0,0,255), thickness = 40)
    cv2.fillPoly(filled, np.int_([pts]), (0,255, 0))
    unwarped_filled = cv2.warpPerspective(filled, Minv, (dst.shape[1], dst.shape[0]))
    dst_filled = cv2.addWeighted(dst, 1, unwarped_filled, 0.5, 0)
    offset = off_center(leftx, rightx, filled.shape[1])
    font = cv2.FONT_HERSHEY_COMPLEX
    
    # curving to the right or left:
    if left_fit[1]>0:
        curve_to = 'the Left'
    else:
        curve_to = 'the Right'
    lane_cur = 'Curving to ' + curve_to + ': ' + str(int(lane_cur)) + 'm'
    cv2.putText(dst_filled, lane_cur, (100,80), font, fontScale = 1.5, color=(0,0,255), thickness = 4)
    cv2.putText(dst_filled, offset, (100,140), font, fontScale = 1.5, color=(0,0,255), thickness = 4)
    
    return dst_filled
