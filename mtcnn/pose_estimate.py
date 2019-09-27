# encoding=utf8  
import os
import numpy as np
import cv2
from test import *
import math
import _pickle as pickle
# from PIL import Image,ImageDraw,ImageFont

data_dir = r'.../test_datasets/'
save_dir = r'.../results/'

def drawResult(imgpath, yaw, pitch, roll,save_dir):
	img = cv2.imread(imgpath)
	draw = img.copy()
	cv2.putText(draw,"Yaw:"+str(yaw),(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
	cv2.putText(draw,"Pitch:"+str(pitch),(20,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
	cv2.putText(draw,"Roll:"+str(roll),(20,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
	cv2.waitKey()
	cv2.imwrite(save_dir+os.path.splitext(imgpath)[0]+'_pose_estimate1.jpg',draw)



def rot2Euler(imgpath,rotation_vector,save_dir):
	# calculate rotation angles
	theta = cv2.norm(rotation_vector, cv2.NORM_L2)
	
	# transformed to quaterniond
	w = math.cos(theta / 2)
	x = math.sin(theta / 2)*rotation_vector[0][0] / theta
	y = math.sin(theta / 2)*rotation_vector[1][0] / theta
	z = math.sin(theta / 2)*rotation_vector[2][0] / theta
	
	ysqr = y * y
	# pitch (x-axis rotation)
	t0 = 2.0 * (w * x + y * z)
	t1 = 1.0 - 2.0 * (x * x + ysqr)
	print('t0:{}, t1:{}'.format(t0, t1))
	pitch = math.atan2(t0, t1) - 0.8356857
	
	# yaw (y-axis rotation)
	t2 = 2.0 * (w * y - z * x)
	if t2 > 1.0:
		t2 = 1.0
	if t2 < -1.0:
		t2 = -1.0
	yaw = math.asin(t2) + 0.005409
	
	# roll (z-axis rotation)
	t3 = 2.0 * (w * z + x * y)
	t4 = 1.0 - 2.0 * (ysqr + z * z)
	roll = math.atan2(t3, t4) - 2.573345436
	
	# 单位转换：将弧度转换为度
	pitch_degree = int((pitch/math.pi)*180)
	yaw_degree = int((yaw/math.pi)*180)
	roll_degree = int((roll/math.pi)*180)
	
	drawResult(imgpath,yaw, pitch, roll,save_dir)
	
	print("Radians:")
	print("Yaw:",yaw)
	print("Pitch:",pitch)
	print("Roll:",roll)
	
	img = cv2.imread(imgpath)
	draw = img.copy()
	cv2.putText(draw,"Yaw:"+str(yaw),(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
	cv2.putText(draw,"Pitch:"+str(pitch),(20,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
	cv2.putText(draw,"Roll:"+str(roll),(20,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
	cv2.waitKey()
	cv2.imwrite(os.path.splitext(imgpath)[0]+'_pose_estimate1.jpg',draw)
	
	print("Degrees:")
	draw = img.copy()
	if yaw_degree > 0:
		output_yaw = "face turns left:"+str(abs(yaw_degree))+" degrees"
		cv2.putText(draw,output_yaw,(20,40),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0))
		print(output_yaw)
	if yaw_degree < 0:
		output_yaw = "face turns right:"+str(abs(yaw_degree))+" degrees"
		cv2.putText(draw,output_yaw,(20,40),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0))
		print(output_yaw)
	if pitch_degree < 0:
		output_pitch = "face downwards:"+str(abs(pitch_degree))+" degrees"
		cv2.putText(draw,output_pitch,(20,80),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0))
		print(output_pitch)
	if pitch_degree > 0:
		output_pitch = "face upwards:"+str(abs(pitch_degree))+" degrees"
		cv2.putText(draw,output_pitch,(20,80),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0))
		print(output_pitch)
	if roll_degree < 0:
		output_roll = "face bends to the right:"+str(abs(roll_degree))+" degrees"
		cv2.putText(draw,output_roll,(20,120),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0))
		print(output_roll)
	if roll_degree > 0:
		output_roll = "face bends to the left:"+str(abs(roll_degree))+" degrees"
		cv2.putText(draw,output_roll,(20,120),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0))
		print(output_roll)
	if abs(yaw) < 0.00001 and abs(pitch) < 0.00001 and abs(roll)< 0.00001:
		cv2.putText(draw,"Initial ststus",(20,40),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0))
		print("Initial ststus")
	cv2.imwrite(save_dir+os.path.splitext(imgpath)[0]+'_pose_estimate2.jpg',draw)


def headPosEstimate(imgpath, landmarks, bbox,save_dir):
	# solvePnP函数的所有输入矩阵必须是double类型 
	# 3D model points
	model_3d_points = np.array(([-165.0, 170.0, -115.0],  # Left eye 
								[165.0, 170.0, -115.0],   # Right eye
								[0.0, 0.0, 0.0],          # Nose tip
								[-150.0, -150.0, -125.0], # Left Mouth corner
								[150.0, -150.0, -125.0]), dtype=np.double) # Right Mouth corner)
	landmarks.dtype = np.double
	# Camera internals
	img = cv2.imread(imgpath)
	img_size = img.shape
	focal_length = img_size[1]
	center =  [img_size[1]/2, img_size[0]/2]
	camera_matrix = np.array(([focal_length, 0, center[0]],
							[0, focal_length, center[1]],
							[0, 0, 1]),dtype=np.double)


	dist_coeffs = np.array([0,0,0,0], dtype=np.double)
	found, rotation_vector, translation_vector = cv2.solvePnP(model_3d_points, landmarks, camera_matrix, dist_coeffs)

	rot2Euler(imgpath,rotation_vector,save_dir)


# 测试图像路径
image_names = os.listdir(data_dir)
for index, image_name in enumerate(image_names):
    imgpath = data_dir + image_name
    bbox, landmarks = get_landmarks(imgpath)
    print("Image_index:", index)
    headPosEstimate(imgpath, landmarks, bbox, save_dir)