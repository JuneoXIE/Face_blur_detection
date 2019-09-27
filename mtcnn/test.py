import sys
sys.path.append('.')
sys.path.append('/home/cmcc/caffe-master/python')
import os
import caffe
import cv2
import numpy as np
import _pickle as pickle
import mtcnn.tools_matrix as tools
dirpath = r"/home/mmlab2/Face_blur_detection/mtcnn"
deploy = os.path.join(dirpath,'12net.prototxt')
caffemodel = os.path.join(dirpath,'12net.caffemodel')
net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = os.path.join(dirpath,'24net.prototxt')
caffemodel = os.path.join(dirpath,'24net.caffemodel')
net_24 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = os.path.join(dirpath,'48net.prototxt')
caffemodel = os.path.join(dirpath,'48net.caffemodel')
net_48 = caffe.Net(deploy,caffemodel,caffe.TEST)


def detectFace(img_path,threshold):
    img = cv2.imread(img_path)
    caffe_img = (img.copy()-127.5)/128
    origin_h,origin_w,ch = caffe_img.shape
    scales = tools.calculateScales(img)
    out = []
    for scale in scales:
        hs = int(origin_h*scale)
        ws = int(origin_w*scale)
        scale_img = cv2.resize(caffe_img,(ws,hs))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_12.blobs['data'].reshape(1,3,ws,hs)
        net_12.blobs['data'].data[...]=scale_img
        # caffe.set_device(0)
        # caffe.set_mode_gpu()
        out_ = net_12.forward()
        out.append(out_)
    image_num = len(scales)
    rectangles = []
    for i in range(image_num):    
        cls_prob = out[i]['prob1'][0][1]
        roi      = out[i]['conv4-2'][0]
        out_h,out_w = cls_prob.shape
        out_side = max(out_h,out_w)
        rectangle = tools.detect_face_12net(cls_prob,roi,out_side,1/scales[i],origin_w,origin_h,threshold[0])
        rectangles.extend(rectangle)
    rectangles = tools.NMS(rectangles,0.7,'iou')

    if len(rectangles)==0:
        return rectangles
    net_24.blobs['data'].reshape(len(rectangles),3,24,24)
    crop_number = 0
    for rectangle in rectangles:
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img,(24,24))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_24.blobs['data'].data[crop_number] =scale_img 
        crop_number += 1
    out = net_24.forward()
    cls_prob = out['prob1']
    roi_prob = out['conv5-2']
    rectangles = tools.filter_face_24net(cls_prob,roi_prob,rectangles,origin_w,origin_h,threshold[1])

    if len(rectangles)==0:
        return rectangles
    net_48.blobs['data'].reshape(len(rectangles),3,48,48)
    crop_number = 0
    for rectangle in rectangles:
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img,(48,48))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_48.blobs['data'].data[crop_number] =scale_img 
        crop_number += 1
    out = net_48.forward()
    cls_prob = out['prob1']
    roi_prob = out['conv6-2']
    pts_prob = out['conv6-3']
    rectangles = tools.filter_face_48net(cls_prob,roi_prob,pts_prob,rectangles,origin_w,origin_h,threshold[2])

    return rectangles

def get_bbox(imgpath):
    threshold = [0.4,0.5,0.6]
    rectangles = detectFace(imgpath,threshold)
    img = cv2.imread(imgpath)
    #draw = img.copy()
    bbox = []
    #landmarks = []
    for rectangle in rectangles:
        #cv2.putText(draw,str(rectangle[4]),(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        #cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
        
        bbox.append((int(rectangle[0]),int(rectangle[1]),int(rectangle[2]),int(rectangle[3]))) # left high right low
        
        #for i in range(5,15,2):
            # cv2.circle(draw,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
            #landmarks.append((int(rectangle[i+0]),int(rectangle[i+1])))

    # cv2.waitKey()
    # cv2.imwrite(os.path.splitext(imgpath)[0]+'_mtcnn_result.jpg',draw)
    # landmarks = np.array(landmarks)
    bbox = np.array(bbox)
    #print(bbox.shape) # 1*4
    return bbox

def save_bbox(imgfilepath):
    bboxes = []
    img_names = os.listdir(imgfilepath)
    for img_name in img_names:
        bbox,_= get_landmarks(imgfilepath + '/' + img_name)
        bboxes.append(bbox)
    bboxes = np.array(bboxes)
    output = open('bboxes.dat', 'wb')
    pickle.dump(bboxes, output)
    output.close()

if __name__ == '__main__':
    imgfilepath = "/xny/mtcnn-caffe-master/demo/test_datasets"
    save_bbox(imgfilepath)