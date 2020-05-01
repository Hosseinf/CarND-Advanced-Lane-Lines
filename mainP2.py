import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import os

from line import Line
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
# from IPython.display import HTML


class Driver():
	def __init__(self):
		self.imgpoints = []
		self.objpoints = []
		self.mtx = []
		self.dist = []

		# polynomial coefficients
		self.left_fit = []
		self.right_fit = []

		self.left_fit_curv = []
		self.right_fit_curv = []

		# list of polynomial coefficients used for averageing the path
		self.left_fit_list = []
		self.right_fit_list = []

		# number of frame that is needed to average through the polynomial coefficients
		self.Nframe = 5

		# counter for number of frame 
		self.frame_num = 0

		# the points of input image and output image where we are finding the unwarped image using prospective transformation matrix
		self.src = []
		self.dst = []

		# was the line detected in the last iteration?
		self.right_detected = False
		self.left_detected = False  

		# curvature of the lines
		self.left_curverad = 0
		self.right_curverad = 0



	def read_img(self,Dir):

		# read the image or frame here
		img = mpimg.imread(Dir)

		print('The dimensions is:', img.shape)

		return img



	def grayscale(self, img):
	    """Applies the Grayscale transform"""
	    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


	def camera_calib(self):

		Dir = r'C:\Users\hosse\Documents\UDACITY\full_self_driving_car\CarND-Advanced-Lane-Lines\camera_cal/'

		image_list = os.listdir(Dir)

		# chessboard size
		nx = 9
		ny = 6

		# create object points (this is the points of real object which in this case is the chaseboard)
		objp = np.zeros((nx*ny,3), np.float32)
		objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

		# print (objp)



		for im in image_list:
			# im = 'calibration2.jpg'
			img = self.read_img(Dir + im)

			gray_scale = self.grayscale(img)

			# find all the corners in the chessboard image
			ret, corners = cv2.findChessboardCorners(gray_scale, (nx,ny), None)

			# print (ret)

			# print (corners)

			if ret==True:
				# draw corners on the chessboard image
				img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

				self.imgpoints.append(corners)
				self.objpoints.append(objp)


				# plt.imshow(img)
				# plt.show()


		# find the transform matirx and distortion coefficient
		ret, self.mtx, self.dist, rvec, tvec = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray_scale.shape[::-1], None, None)

		print(self.mtx)
		print (self.dist)


	def undistort(self, img):
		# undistort the image
		undist_img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

		# plt.imshow(undist_img)
		# plt.show()

		return undist_img



	def gradient_thresh(self,img,grad_dir,thresh_low,thresh_high,kernel_size):

		# convert to gray-scale
		gray = self.grayscale(img)

		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
		sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,  ksize=kernel_size)
		if grad_dir == 'x':
			sobel = sobelx
		elif grad_dir == 'y':
			sobel = sobely
		if grad_dir == 'mag':
			sobel = np.sqrt(sobelx**2 + sobely**2)
		if grad_dir == 'dir':
			sobel = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

		abs_sobel = np.absolute(sobel)
		if grad_dir != 'dir':
			scale_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
		else:
			scale_sobel = abs_sobel

		binary_img = np.zeros_like(gray)
		binary_img[(scale_sobel>thresh_low) & (scale_sobel<thresh_high)] = 1

		return binary_img,scale_sobel



	def color_thresh(self,img, thresh_low,thresh_high):
		hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

		H = hls[:,:,0]
		L = hls[:,:,1]
		S = hls[:,:,2]

		hsv_thresh=(200, 255)

		binary_img = np.zeros_like(S)

		# binary_img[(S>thresh_low) & (S<thresh_high) ] = 1

		binary_img[(L>170) & (L<255) ] = 1


		# Convert BGR to HSV
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		# define range of yellow color in HSV
		lower_yellow = np.array([20, 100, 100])
		upper_yellow = np.array([100, 255, 255])

		# Threshold the HSV image to get only yellow colors
		yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)


		output = np.zeros_like(yellow_mask)
		output[(yellow_mask != 0) | (binary_img == 1)] = 1

		return output, S


	def cobined_grad_color_thresh(self, img, grad_dir,grad_thresh_low,grad_thresh_high,kernel_size, color_thresh_low, color_thresh_high):

		# use gradient threshold here
		b,scale_sobel = self.gradient_thresh(img, grad_dir, grad_thresh_low, grad_thresh_high, kernel_size)

		b,scale_sobel_d = self.gradient_thresh(img, 'dir', np.pi/6,np.pi/3, kernel_size)

		

		b,S = self.color_thresh(img, color_thresh_low, color_thresh_high)

		binary_img = np.zeros_like(S)

		binary_img[((S>color_thresh_low) & (S<color_thresh_high)) | ((scale_sobel>grad_thresh_low) & (scale_sobel<grad_thresh_high)) ] = 1
		# & (((scale_sobel_d>np.pi/6) & (scale_sobel_d<np.pi/3)))

		return binary_img


	def perspective(self, img):

		# find the perspective transform matrix
		M = cv2.getPerspectiveTransform(self.src, self.dst)

		warped_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

		return warped_img


	def hist(self, img):
		# only consider the bottom half of the image
		img_size = img.shape
		bottom_half = int(img_size[0]/2)

		histogram = np.sum(img[bottom_half:,:], axis=0)

		# plt.figure()
		# plt.plot(histogram)

		# find the max points at left and right side

		# find the mid point
		mid_point = int(img_size[1]/2)

		print ('mid_point=', mid_point)

		# find the max index for both sides
		left_max_ind = np.argmax(histogram[:mid_point])
		right_max_ind = mid_point + np.argmax(histogram[mid_point:])

		print ('histogram size= ', len(histogram))
		print ('left_max_ind= ', left_max_ind)
		print ('right_max_ind= ', right_max_ind)

		return left_max_ind, right_max_ind


	def line_detection(self, img, left_max_ind, right_max_ind):

		# setup the windows for detecting lines
		# number of windows
		nwindows = 9

		# width of windows by +/- margin
		margin = 100

		# min of number of pixesl in the window
		minPix = 50

		# height of windows
		win_height = int(img.shape[0]/nwindows)

		# identify the nonzero element in the image
		nonzero = img.nonzero()
		nonzerox = np.array(nonzero[1])
		nonzeroy = np.array(nonzero[0])

		left_current_idx = left_max_ind
		right_current_idx = right_max_ind


		# define list for right and left windows to keep the index
		left_lane_idx = []
		right_lane_idx = []

		out_img = np.dstack((img, img, img))*255

		# find all the windows
		for win in range(nwindows):
			win_y_low = img.shape[0] - (win+1)*win_height
			win_y_high = img.shape[0] - (win)*win_height

			left_win_x_low = left_current_idx - margin
			left_win_x_high = left_current_idx + margin

			right_win_x_low = right_current_idx - margin
			right_win_x_high = right_current_idx + margin

			# Draw the windows on the visualization image
			cv2.rectangle(out_img,(left_win_x_low,win_y_low),
			(left_win_x_high,win_y_high),(0,255,0), 2) 
			cv2.rectangle(out_img,(right_win_x_low,win_y_low),
			(right_win_x_high,win_y_high),(0,255,0), 2) 

			# find the nonzero pixel in each windows
			win_left_idx = ((nonzerox>left_win_x_low) & (nonzerox<left_win_x_high) & (nonzeroy>win_y_low) & (nonzeroy<win_y_high)).nonzero()[0]
			win_right_idx = ((nonzerox>right_win_x_low) & (nonzerox<right_win_x_high) & (nonzeroy>win_y_low) & (nonzeroy<win_y_high)).nonzero()[0]


			# print('win_left_idx=',win_left_idx)
			# print('nonzerox=',nonzerox[win_left_idx])

			# print ('(nonzerox>left_win_x_low) =', (nonzerox>left_win_x_low).nonzero()[0])

			# print ('(nonzerox>left_win_x_low) ======', nonzerox[(nonzerox>left_win_x_low).nonzero()[0]])


			# add the pixel of each window to the list
			left_lane_idx.append(win_left_idx)
			right_lane_idx.append(win_right_idx)


			# adjust the window with respect to the density of the pixels
			if len(win_left_idx) > minPix:
				left_current_idx = int(np.mean(nonzerox[win_left_idx]))

			if len(win_right_idx) > minPix:
				right_current_idx = int(np.mean(nonzerox[win_right_idx]))



		left_lane_idx = np.concatenate(left_lane_idx)
		right_lane_idx = np.concatenate(right_lane_idx)


		leftx = nonzerox[left_lane_idx]
		lefty = nonzeroy[left_lane_idx]
		rightx = nonzerox[right_lane_idx]
		righty = nonzeroy[right_lane_idx]


		out_img[lefty, leftx] = [255,0,0]
		out_img[righty, rightx] = [0,0,255]
		# out_img[:,:,1] = out_img[:,:,1]*0

		if (len(leftx)>0 and len(lefty)>0):
			self.left_detected = True
		else:
			self.left_detected = False

		if (len(rightx)>0 and len(righty)>0):
			self.right_detected = True
		else:
			self.right_detected = False


		# plt.imshow(out_img, cmap='gray')
		# plt.show()


		return leftx, lefty, rightx, righty, out_img

	def fit_ploy(self, img, leftx, lefty, rightx, righty):

		# generate the polynomial coefficient
		if (len(leftx)>5 or len(lefty)>5):
			left_fit = np.polyfit(lefty, leftx, 2)
		else:
			left_fit = self.left_fit
		if (len(rightx)>5 or len(righty)>5):
			right_fit = np.polyfit(righty, rightx, 2)
		else:
			right_fit = self.right_fit

		left_curverad, right_curverad = self.measure_curvature_pixels(leftx, lefty, rightx, righty, img)

		print ('right_curverad=',right_curverad)
		print ('left_curverad=',left_curverad)

		# update the polynomial coefficient if it is reasonable
		if (np.abs(np.abs(left_curverad) - np.abs(self.left_curverad)) < 300) and (self.left_curverad != 0):
			self.left_fit = left_fit
			self.left_curverad = left_curverad
			print ('left lane is updated')

		if self.left_curverad == 0:
			self.left_fit = left_fit
			self.left_curverad = left_curverad
			print ('left lane is updated only once')


		if (np.abs(np.abs(right_curverad) - np.abs(self.right_curverad)) < 600) and (self.right_curverad != 0):
			self.right_fit = right_fit
			self.right_curverad = right_curverad

		if self.right_curverad == 0:
			self.right_fit = right_fit
			self.right_curverad = right_curverad


		# smoothing the results
		if len(self.left_fit_list) == self.Nframe:
			print ('averaging the coefficient of the polynomial ...')

			left_fit_ave = np.sum(self.left_fit_list, axis=0)/self.Nframe
			right_fit_ave = np.sum(self.right_fit_list, axis=0)/self.Nframe

			print('left_fit_ave=', left_fit_ave)

			# generate the equation
			left_fitx = np.poly1d(left_fit_ave)
			right_fitx = np.poly1d(right_fit_ave)

			# remove the first element of the list
			self.left_fit_list.pop(0)
			self.right_fit_list.pop(0)


		else:
			# generate the equation
			left_fitx = np.poly1d(self.left_fit)
			right_fitx = np.poly1d(self.right_fit)

		# generate the y values
		yValue = np.linspace(0, img.shape[0]-1, img.shape[0])

		# xValue_left = left_fit[0]*yValue**2 + left_fit[1]*yValue + left_fit[2]
		# xValue_right = right_fit[0]*yValue**2 + right_fit[1]*yValue + right_fit[2]

		
		# evaluate the x value
		xValue_left = left_fitx(yValue)
		xValue_right = right_fitx(yValue)

		# plt.plot(xValue_left, yValue, color ='yellow')
		# plt.plot(xValue_right, yValue, color ='yellow')

		# plt.show()

		return xValue_left, xValue_right, yValue



	def search_around_poly(self, img ):

		nonzero = img.nonzero()
		nonzerox = np.array(nonzero[1])
		nonzeroy = np.array(nonzero[0])

		# lane margin
		margin = 100

		left_fitx = self.left_fit[0]*nonzeroy**2 + self.left_fit[1]*nonzeroy + self.left_fit[2]
		right_fitx = self.right_fit[0]*nonzeroy**2 + self.right_fit[1]*nonzeroy + self.right_fit[2]

		# find the lane index between the boundry
		left_lane_idx = ((nonzerox >= left_fitx-margin) & (nonzerox <= left_fitx+margin)).nonzero()[0]
		right_lane_idx = ((nonzerox >= right_fitx-margin) & (nonzerox <= right_fitx+margin)).nonzero()[0]


		leftx = nonzerox[left_lane_idx]
		lefty = nonzeroy[left_lane_idx]

		rightx = nonzerox[right_lane_idx]
		righty = nonzeroy[right_lane_idx]

		xValue_left, xValue_right, yValue = self.fit_ploy(img, leftx, lefty, rightx, righty)


		# visualization
		out_img = np.dstack((img, img, img))*255

		out_img[lefty, leftx] = [255,0,0]
		out_img[righty, rightx] = [0,0,255]

		left_line_window1 = np.array([np.transpose([xValue_left-margin, yValue])])
		left_line_window2 = np.array([np.flipud(np.transpose([xValue_left+margin, 
                              yValue]))])

		left_line_pts = np.hstack((left_line_window1, left_line_window2))


		right_line_window1 = np.array([np.transpose([xValue_right-margin, yValue])])
		right_line_window2 = np.array([np.flipud(np.transpose([xValue_right+margin, 
                              yValue]))])

		right_line_pts = np.hstack((right_line_window1, right_line_window2))

		# draw the polygon to identify the area where the line could be
		window_img = np.zeros_like(out_img)
		cv2.fillPoly(window_img, np.int_(left_line_pts), [0,255,0])
		cv2.fillPoly(window_img, np.int_(right_line_pts), [0,255,0])

		final_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

		# plt.plot(xValue_left, yValue, color ='yellow')
		# plt.plot(xValue_right, yValue, color ='yellow')


		# measure the curvature of the lane lines
		# left_curverad, right_curverad = self.measure_curvature_pixels(leftx, lefty, rightx, righty, img)
		# print ('right_curverad=',right_curverad)
		# print ('left_curverad=',left_curverad)

		# print(left_line_pts)
		# plt.imshow(final_img, cmap='gray')
		# plt.show()


		if (len(leftx)>0 and len(lefty)>0):
			self.left_detected = True
			self.left_fit_list.append(self.left_fit)
		else:
			self.left_detected = False

		if (len(rightx)>0 and len(righty)>0):
			self.right_detected = True
			self.right_fit_list.append(self.right_fit)
		else:
			self.right_detected = False

		print (self.left_fit_list)

		return final_img, xValue_left, xValue_right, yValue


	def measure_curvature_pixels(self, leftx, lefty, rightx, righty, img):

		img_size = img.shape

		# calculate the histogram
		left_ind, right_ind = self.hist(img)
		diff_pixel = right_ind - left_ind
		print('diff_pixel=',diff_pixel)

		ym_per_pix = 30/img_size[0]  # meters per pixel in y dimension
		xm_per_pix = 3.7/diff_pixel # meters per pixel in x dimension

		# calculate the polynomial based on the meter values
		# generate the polynomial coefficient
		if (len(leftx)>5 or len(lefty)>5):
			left_fit = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
		else:
			left_fit = self.left_fit_curv
		if (len(rightx)>5 or len(righty)>5):
			right_fit = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
		else:
			right_fit = self.right_fit_curv


		# generate the equation
		# left_fitx = np.poly1d(left_fit)
		# right_fitx = np.poly1d(right_fit)

		# generate the y values
		yValue = np.linspace(0, img.shape[0]-1, img.shape[0])



		# find the maximum y point
		yMax = np.max(yValue*ym_per_pix)

		# calculate the curvature for each side
		left_curverad = ((1 + (2*yMax*left_fit[0] + left_fit[1])**2)**(3/2)) / (2*left_fit[0])
		right_curverad = ((1 + (2*yMax*right_fit[0] + right_fit[1])**2)**(3/2)) / (2*right_fit[0])

		# update the global variables
		self.left_fit_curv = left_fit
		self.right_fit_curv = right_fit


		return left_curverad, right_curverad



	def project_lane_line(self, img, original, xValue_left, xValue_right, yValue):


		# Create an image to draw the lines on
		warp_zero = np.zeros_like(img[:,:,0]).astype(np.uint8)
		color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

		# Recast the x and y points into usable format for cv2.fillPoly()
		pts_left = np.array([np.transpose(np.vstack([xValue_left, yValue]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([xValue_right, yValue])))])
		pts = np.hstack((pts_left, pts_right))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(color_warp, np.int_(pts), (0,255, 0))

		# find the perspective transform matrix
		Minv = cv2.getPerspectiveTransform(self.dst, self.src)

		newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


		result = cv2.addWeighted(original, 1, newwarp, 0.3, 0)

		return result


	def line_drawer(self, img):

		original = np.copy(img)

		self.frame_num += 1
		print ('self.frame_num = ', self.frame_num)

		# plt.imshow(img, cmap='gray')
		# plt.show()


		offsetx = 300
		offsety = 20
		h,w,c = img.shape
		# self.src = np.float32([[img.shape[1]/2.11, img.shape[0]/1.64], [img.shape[1]/4.98, img.shape[0]/1.07], [img.shape[1]/1.21, img.shape[0]/1.07], [img.shape[1]/1.899, img.shape[0]/1.64]])
		# # self.src = np.float32([[img.shape[1]/1.97, img.shape[0]/1.55], [img.shape[1]/4.29, img.shape[0]/1.05], [img.shape[1]/1.18, img.shape[0]/1.05], [img.shape[1]/1.787, img.shape[0]/1.55]])
		self.dst = np.float32([[offsetx,offsety], [offsetx, img.shape[0]-offsety], [img.shape[1]-offsetx, img.shape[0]-offsety], [img.shape[1]-offsetx, offsety]])


		self.src = np.float32([(575,464),
		                  (258,682), 
		                  (1049,682),
		                  (707,464)])
		# self.dst = np.float32([(450,0),
		#                   (w-450,0),
		#                   (450,h),
		#                   (w-450,h)])

		# Dir = r'C:\Users\hosse\Documents\UDACITY\full_self_driving_car\CarND-Advanced-Lane-Lines\camera_cal/'
		# im = 'calibration2.jpg'
		# img = driver.read_img(Dir + im)

		# undistort the image
		img = driver.undistort(img)

		print(img.shape)

		# plt.imshow(img, cmap='gray')
		# plt.show()

		# apply gradient threshold to the image
		# img,b = driver.gradient_thresh(img,'x',50,200, 9)

		# apply the color threshold to the image
		img, s = driver.color_thresh(img,100,255)

		# apply the prespective to the image
		img = self.perspective(img)

		# apply the cobined threshold for color and grad
		# img = self.cobined_grad_color_thresh(img, 'x',50,200, 9, 100,255)
		# img = self.cobined_grad_color_thresh(img, 'x',50,240, 9, 100,255)

		# img,b = driver.gradient_thresh(img,'dir',np.pi/6,np.pi/3, 9)
		# img,b = driver.gradient_thresh(img,'x',50,200, 9)
		# img, b = driver.color_thresh(img,100,255)

		# plt.imshow(img, cmap='gray')
		# plt.show()


		if (self.right_detected == False) or (self.left_detected == False):
			# find the approximate position of the lane lines
			left_max_ind, right_max_ind = self.hist(img)
			# find the left and right lane's points
			leftx, lefty, rightx, righty, img3 = self.line_detection(img, left_max_ind, right_max_ind)
			# fit a polynomial on the detected points
			xValue_left, xValue_right, yValue = self.fit_ploy(img, leftx, lefty, rightx, righty)

		else:

			print('****************************************************************')
			img3, xValue_left, xValue_right, yValue = self.search_around_poly(img)


		# # find the approximate position of the lane lines
		# left_max_ind, right_max_ind = self.hist(img)
		# # find the left and right lane's points
		# leftx, lefty, rightx, righty, img3 = self.line_detection(img, left_max_ind, right_max_ind)
		# # fit a polynomial on the detected points
		# xValue_left, xValue_right, yValue = self.fit_ploy(img, leftx, lefty, rightx, righty)

		img3 = self.project_lane_line(img3, original, xValue_left, xValue_right, yValue)



		# print (img.shape)
		# plt.imshow(img3, cmap='gray')
		# plt.show()

		return img3



# initialize the code here
line = Line()
print(line.detected)
driver = Driver()

Dir = r'C:\Users\hosse\Documents\UDACITY\full_self_driving_car\CarND-Advanced-Lane-Lines\test_images/'

save_Dir = r'C:\Users\hosse\Documents\UDACITY\full_self_driving_car\CarND-Advanced-Lane-Lines\output_images/'

image_list = os.listdir(Dir)

# calibrate the camera
driver.camera_calib()

# for im in image_list:
# 	img = driver.read_img(Dir + im)

# 	img = driver.line_drawer(img)

# 	plt.imshow(img, cmap='gray')
# 	plt.show()

# 	image_name = save_Dir+im+'.jpg'
# 	cv2.imwrite(image_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# example image
im = 'test2.jpg'
img = driver.read_img(Dir + im)
img = driver.line_drawer(img)


# run the video frame
video_Dir = r'C:\Users\hosse\Documents\UDACITY\full_self_driving_car\CarND-Advanced-Lane-Lines/'

# capture the video
cap = cv2.VideoCapture(video_Dir+'test_videos\challenge_video.mp4')

## create an object to generate a video from the frame
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output video path
VideoOut = video_Dir + 'test_video_output/videos/challenge_video.avi'
# frame rate
fps = 15
# find the frame size
ret, frame = cap.read()
height, width, layers = frame.shape
size = (width,height)
out = cv2.VideoWriter(VideoOut,fourcc, fps, size)

i=0
while(cap.isOpened()):
    ret, frame = cap.read()

    # plt.imshow(frame, cmap='gray')
    # plt.show()
    if ret == False:
        break
    print ('frame size = ', frame.shape)
    if i % 1 == 0:

    	frame = driver.line_drawer(frame)

    	# save each frame of the video
    	# save_dist = video_Dir + 'test_video_output/'+'frame'+str(i)+'.png'
    	# cv2.imwrite(save_dist,frame)

    	# add all the frame to the list for creating a video
    	out.write(frame)

    i+=1



out.release()
cap.release()
cv2.destroyAllWindows()


