# -*-coding=UTF-8-*-
"""
在无参考图下，检测图片质量的方法
"""
import os
import cv2
import numpy as np
import mtcnn.tools_matrix as tools
from mtcnn.test import get_bbox
from skimage.transform import estimate_transform, warp
from skimage import filters
import math

class BlurScore:
    def __init__(self, strDir):
        """
        创建BlurScore对象
        :param: strDir 存放测试图像的文件夹路径
        :return:  BlurScore对象
        """
        print("BlurScore object is created...")
        self.strDir = strDir # strDir：保存测试图像的文件夹路径

    def _getAllImg(self):
        """
        根据目录读取所有的图片
        :return:  图片列表
        """
        names = []
        for root, dirs, files in os.walk(self.strDir):  
            for file in files:
                names.append(str(file))
        return names


    def _imageToMatrix(self, img):
        """
        将图片对象转化矩阵
        :param img: 图像对象
        :return imgMat: 返回矩阵
        """
        imgMat = np.matrix(img)
        return imgMat


    def _cropImgByMtcnn(self,imgName):
        """
        图像的读取及预处理操作
        :param imgName: 图像的名称
        :return: 剪裁人脸后的图像
        """
        strPath = os.path.join(self.strDir, imgName)
        img = cv2.imread(strPath)
        bbox = get_bbox(strPath)
        left = bbox[0,0]
        right = bbox[0,2]
        top = bbox[0,1]
        bottom = bbox[0,3]
        cropped_image = img[int(top):int(bottom), int(left):int(right)]
        
        return cropped_image


    def _preprocess(self, imgName, do_gray = True):
        """
        图像的读取及预处理操作
        :param imgName: 图像的名称
        :return: 经过剪裁人脸及灰度化之后的图片对象
        """
        # 剪裁处理
        cropImg = self._cropImgByMtcnn(imgName)
        
        # 将图片压缩为单通道的灰度图（可选）
        if do_gray:
            img2gray = cv2.cvtColor(cropImg.astype('float32'), cv2.COLOR_BGR2GRAY)
            
            return img2gray, cropImg
        
        return cropImg

    
    def getAllScore(self):
        """
        对整个数据集进行处理，把所有图像11个评价指标的结果记录在txt文件中
        :return: result.txt
        """
        names = self._getAllImg()
        f1 = open(os.path.join(r"/home/mmlab2/Face_blur_detection",'result.txt'),'w+')
        # f1.write("Image name    Brenner    Energy    Entropy    Laplacian    SMD    SMD2    Thenengrad    Variance    Vollath    JPEG    JPEG2\n")
        f1.write("Gaussian_laplacian  \n")
        for index, name in enumerate(names):
            print("Processing image: ", index+1)
            f1.write(str(name)+"  ")
            '''
            f1.write(str(self._Brenner(name))+"  ")
            f1.write(str(self._Energy(name))+"  ")
            f1.write(str(self._Entropy(name))+"  ")
            f1.write(str(self._Laplacian(name))+"  ")
            f1.write(str(self._SMD(name))+"  ")
            f1.write(str(self._SMD2(name))+"  ")
            f1.write(str(self._Thenengrad(name))+"  ")
            f1.write(str(self._Variance(name))+"  ")
            f1.write(str(self._Vollath(name))+"  ")
            f1.write(str(self._JPEG(name))+"  ")
            f1.write(str(self._JPEG2(name)))
            f1.write("  \n")
            '''
            f1.write(str(self._Gaussian_Laplacian(name)))
            f1.write("\n")
        f1.close()
        return 


    def _Brenner(self, imgName):
        """
        指标一：Brenner梯度函数
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        img2gray, cropImg = self._preprocess(imgName, do_gray = True)
        #cv2.imwrite(r"/home/mmlab2/Face_blur_detection/"+str(imgName)+"_crop.jpg",cropImg,[int(cv2.IMWRITE_JPEG_QUALITY),70])
        f = self._imageToMatrix(img2gray)
        x, y = f.shape
        score = 0
        for i in range(x-2):
            for j in range(y-2):
                score += (f[i+2, j] - f[i, j])**2
        return score


    def _Laplacian(self, imgName):
        """
        指标二：拉普拉斯方差算法
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        cropImg = self._preprocess(imgName, do_gray = False)
        score = cv2.Laplacian(cropImg, cv2.CV_64F).var()
        return score


    def _Thenengrad(self, imgName):
        """
        指标三：Thenengrad梯度函数
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        img2gray, cropImg = self._preprocess(imgName, do_gray = True)
        f = self._imageToMatrix(img2gray)

        tmp = filters.sobel(f)
        score = np.sum(tmp**2)
        score = np.sqrt(score)
        return score
    

    def _SMD(self, imgName):
        """
        指标四：SMD灰度方差函数
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        # 图像的预处理
        img2gray, cropImg = self._preprocess(imgName, do_gray = True)
        f = self._imageToMatrix(img2gray)/255.0
        x, y = f.shape
        score = 0
        for i in range(x - 1):
            for j in range(y - 1):
                score += np.abs(f[i+1,j]-f[i,j])+np.abs(f[i,j]-f[i+1,j])
        
        return score
        
        
    def _SMD2(self, imgName):
        """
        指标五：SMD2灰度方差函数
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        # 图像的预处理
        img2gray, cropImg = self._preprocess(imgName, do_gray = True)
        f = self._imageToMatrix(img2gray)/255.0
        x, y = f.shape
        score = 0
        for i in range(x - 1):
            for j in range(y - 1):
                score += np.abs(f[i+1,j]-f[i,j])*np.abs(f[i,j]-f[i,j+1])
        
        return score
        
    
    def _Variance(self, imgName):
        """
        指标六：方差函数
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        # 图像的预处理
        img2gray, cropImg = self._preprocess(imgName, do_gray = True)
        f = self._imageToMatrix(img2gray)
        score = np.var(f)
        
        return score
        
        
    def _Energy(self, imgName):
        """
        指标七：能量梯度函数
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        img2gray, cropImg = self._preprocess(imgName, do_gray = True)
        f = self._imageToMatrix(img2gray)
        x, y = f.shape
        score = 0
        for i in range(0, x-1):
            for j in range(0, y-1):
                score += (f[i+1, j] - f[i, j])**2 * (f[i,j + 1] - f[i,j])**2
                
        return score
    
    
    def _Vollath(self, imgName):
        """
        指标八：Vollath函数
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        img2gray, cropImg = self._preprocess(imgName, do_gray = True)
        f = self._imageToMatrix(img2gray)
        score = 0
        x, y = f.shape
        for i in range(x-1):
            for j in range(y):
                score += f[i,j]*f[i+1,j]
        score = score - x * y * np.mean(f)
       
        return score
        
    
    def _Entropy(self, imgName):
        """
        指标九：熵函数
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        img2gray, cropImg = self._preprocess(imgName, do_gray = True)
        f = np.array(img2gray,dtype = 'int64')
        x, y = f.shape
        count = x*y
        p = np.bincount(f.flatten())
        score = 0
        for i in range(0, len(p)):
            if p[i]!=0:
                score -= p[i]*math.log(p[i]/count)/count
        
        return score
        
    
    def _JPEG(self, imgName):
        """
        指标十：论文No-Reference Perceptual Quality Assessment of JPEG Compressed Images
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        img2gray, cropImg = self._preprocess(imgName, do_gray = True)
        f = np.array(img2gray)
        x,y = f.shape
        # 水平方向
        dh = np.zeros((x,y-1))
        for i in range(x):
            for j in range(y-1):
                dh[i,j] = f[i,j+1] - f[i,j]
        bh = 0
        for i in range(x):
            for j in range(int(y/8)-1):
                bh += abs(dh[i,8*j])
        bh /= x * (int(y/8)-1)
        
        ah = 0
        for i in range(x):
            for j in range(y-1):
                ah += abs(dh[i,j])
        ah = (1/7) * ((8/(x*(y-1))) * ah - bh)
        
        zh = np.zeros((x,y-2))
        for i in range(x):
            for j in range(y-2):
                if dh[i,j] * dh[i,j+1] < 0:
                    zh[i,j] = 1
                else:
                    zh[i,j] = 0
        
        Zh = 0
        for i in range(x):
            for j in range(y-2):
                Zh += zh[i,j]
        Zh /= x*(y-2)
        
        # 垂直方向
        dv = np.zeros((x-1,y))
        for i in range(x-1):
            for j in range(y):
                dv[i,j] = f[i+1,j] - f[i,j]
        bv = 0
        for i in range(int(x/8)-1):
            for j in range(y):
                bv += abs(dv[8*i,j])
        bv /= y * (int(x/8)-1)
        
        av = 0
        for i in range(x-1):
            for j in range(y):
                av += abs(dv[i,j])
        av = (1/7) * ((8/((x-1)*y)) * av - bv)
        
        zv = np.zeros((x-2,y))
        for i in range(x-2):
            for j in range(y):
                if dv[i,j] * dv[i+1,j] < 0:
                    zv[i,j] = 1
                else:
                    zv[i,j] = 0
        
        Zv = 0
        for i in range(x-2):
            for j in range(y):
                Zv += zv[i,j]
        Zv /= y*(x-2)
        
        # 汇总
        B = (bh + bv)/2
        A = (ah + av)/2
        Z = (Zh + Zv)/2
        S = -245.9 + 261.9 * pow(B,-0.024) * pow(A,0.016) * pow(Z, 0.0064)
        
        return S
        
        
    def _JPEG2(self, imgName):
        """
        指标十一：论文No-Reference Image Quality Assessment forJPEG/JPEG2000 Coding
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        S = self._JPEG(imgName)
        SS = 1 + 4 / (1 + math.exp((-1.0217) * (S-3)))
    
        return SS
    
    def _Gaussian_Laplacian(self, imgName):
        '''
        指标十二：
        对采集到的人脸图像进行如下处理：
        1.高斯模糊去噪，
        2.转换灰度图，
        3.在此图像上利用拉普拉斯算子滤波，
        4.直方图归一化映射到0-255，
        5.求均值方差，方差的阈值为300
        '''
        strPath = os.path.join(self.strDir, imgName)
        img = cv2.imread(strPath)
        #img = self._preprocess(imgName, do_gray = False)
        # 高斯滤波
        gauss_blur = cv2.GaussianBlur(img,(3,3),0)
        # 使用线性变换转换输入数组元素成8位无符号整型 归一化为0-255
        transform = cv2.convertScaleAbs(gauss_blur)
        # 灰度化
        grey = cv2.cvtColor(transform, cv2.COLOR_BGR2GRAY)
        # 使用3x3的Laplacian算子卷积滤波
        grey_laplace = cv2.Laplacian(grey, cv2.CV_16S,ksize=3)
        # 归一化为0-255
        resultImg = cv2.convertScaleAbs(grey_laplace)
        
        # 计算均值和方差
        mean, std = cv2.meanStdDev(resultImg)
        blurPer = std[0][0] ** 2
        
        return blurPer
        
if __name__ == "__main__":
    BlurScore = BlurScore(strDir=r"/home/mmlab2/Face_blur_detection/isblur")
    BlurScore.getAllScore() 