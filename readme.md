# 11 operators for face blur detection

## 1.Variance of the Laplacian

```python
import cv2
imagePath ='xxx.jpg'
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print('y1 blur:',cv2.Laplacian(gray, cv2.CV_64F).var())
```



## 2. Brenner Gradient Function

This function calculates the square of the gray difference between two adjacent pixels. The image clarity measure based on Brenner gradient is defined as follows:
$$
D(f)=\sum_{y}\sum_{x}{|f(x+2,7)-f(x,y)|^2}
$$
where f(x,y) represents the of pixel (x,y).

```python
def Brenner(img):
    x, y = img.shape
    D = 0
    for i in range(x-2):
        for j in range(y-2):
            D += (img[i+2, j] - img[i, j])**2
    return D
```



## 3. Tenengrad Gradient Function

This function employs Sobel operator to extract horizontal and vertical gradients of the image. The image clarity measure based on Tenengrad gradient is defined as follows:
$$
D(f)=\sum_{y}\sum_{x}{|G(x,y)|}
$$
G(x,y) is formulated as:
$$
G(x,y)=\sqrt{G_x^2(x,y)+G_y^2(x,y)},G(x,y)>T
$$
where T is the pre-defined edge detection threshold, Gx(x,y) is the convolution results of the horizontal Sobel operator with the image at the pixel (x,y), Gy(x,y) is the convolution results of the vertical Sobel operator with the image at the pixel (x,y).

The employed horizontal and vertical Sobel operators are shown as follows:
$$
g_x= \frac{1}{4}\left [ \begin{matrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{matrix} \right],
g_y=\frac{1}{4}\left [ \begin{matrix}
1 & 2 & 1 \\
0 & 0 & 0 \\
-1 & -2 & -1
\end{matrix} \right]
$$
```python
def Thenengrad(img):
    T = 50 # edge detection threshold
    Grad_value = 0
    Sx, Sy = 0
    x, y = img.shape
    for i in range(1,x-1):
    current_ptr = img[i,:] # 当前行
    pre_ptr = img[i-1,:] # 上一行
    next_ptr= img[i+1,:] # 下一行
    for j in range(1,y-1):
    Sx =  pre_ptr[j - 1] * (-1) + pre_ptr[j + 1] + current_ptr[j - 1] * (-2) + current_ptr[j + 1] * 2 + next_ptr[j - 1] * (-1) + next_ptr[j + 1]; # horizontal gradient
    Sy = pre_ptr[j - 1] + 2 * pre_ptr[j] + pre_ptr[j + 1] - next_ptr[j - 1] - 2 * next_ptr[j] - next_ptr[j + 1] # vertical gradient
    # sum
    G = sqrt(Sx * Sx + Sy * Sy)
    if G > T:
    Grad_value += G
    return Grad_value
```



## 4. SMD (Grey Variance) Function

When the image is fully focused, it is the clearest, containing many high-frequency components. Therefore, the change of grey level can be used as the basis for focusing evaluation. The formula of gray variance method is as follows:
$$
D(f)=\sum_{y}\sum_{x}{|f(x,y)-f(x,y-1)|+|f(x,y)-f(x+1,y)|}
$$

```python
def SMD(img):
    reImg = cv2.resize(img, (800, 900), interpolation=cv2.INTER_CUBIC)  
    img2gray = cv2.cvtColor(reImg, cv2.COLOR_BGR2GRAY) 
    f=self._imageToMatrix(img2gray)/255.0
    x, y = f.shape
    D = 0
    for i in range(x - 1):
    for j in range(y - 1):
    D += np.abs(f[i+1,j]-f[i,j])+np.abs(f[i,j]-f[i+1,j])

    return D
```



## 5.SMD2 Function

The gray difference evaluation function has good computational performance, but its disadvantage is also obvious, that is, it is less sensitive in the areas near the focus, as the function is too flat near the extreme point. A new evaluation function, called gray variance product, is proposed, which multiplies two gray differences of each pixel and then accumulates them one by one. The definition of this function is as follows:
$$
D(f)=\sum_{y}\sum_{x}{|f(x,y)-f(x+1,y)|*|f(x,y)-f(x,y+1)|}
$$

```python
 def SMD2(img):
    reImg = cv2.resize(img, (800, 900), interpolation=cv2.INTER_CUBIC)  
    img2gray = cv2.cvtColor(reImg, cv2.COLOR_BGR2GRAY) 
    f=self._imageToMatrix(img2gray)/255.0
    x, y = f.shape
    D = 0
    for i in range(x - 1):
    for j in range(y - 1):
    D += np.abs(f[i+1,j]-f[i,j])*np.abs(f[i,j]-f[i,j+1])
    return D
```



## 6. Variance Function

$$
D(f)=\sum_{y}\sum_{x}{|f(x,y)-\mu|^2},\mu\ is\ the\ average\ grey\ level\ of\ the\ image
$$

```python
def _SMD2Detection(img):
    reImg = cv2.resize(img, (800, 900), interpolation=cv2.INTER_CUBIC)  
    img2gray = cv2.cvtColor(reImg, cv2.COLOR_BGR2GRAY) 
    f = self._imageToMatrix(img2gray)
    D = np.var(f)

    return D
```



## 7. Energy Gradient Function

The energy gradient function is more suitable for real-time evaluation of image sharpness. The definition of the function is as follows:
$$
D(f)=\sum_{y}\sum_{x}{(|f(x+1,y)-f(x,y)|^2+|f(x,y+1)-f(x,y)|^2)}
$$
M and N are image width and height, respectively.

## 8. Vollath Function

Vollath Function is defined as follows:
$$
D(f)=\sum_{y}\sum_{x}{f(x,y)*f(x+1,y)}-M*N*\mu^2
$$

```python
def Vollath(img):
    # 图像的预处理
    reImg = cv2.resize(img, (800, 900), interpolation=cv2.INTER_CUBIC)  
    img2gray = cv2.cvtColor(reImg, cv2.COLOR_BGR2GRAY)  #
    f = self._imageToMatrix(img2gray)
    D = 0
    x,y=f.shape
    for i in range(x-1):
    for j in range(y):
    D += f[i,j]*f[i+1,j]
    D = D - x * y * np.mean(f)

    return D
```



## 9. Entropy Function

Entropy function based on statistical features is an important operator to measure the information richness of an image. According to information theory, the information quantity of an image f is measured by the information entropy D of the image:
$$
D(f)=-\sum_{i=0}^{L-1}{p_iln(p_i)}
$$
where pi is the probability of pixels with gray value of i in the image, L is the total number of gray levels (usually 256). The larger the value of  D is, the clearer the image is. The entropy function is not very sensitive, with the different content of the image, it is easy to appear the opposite result to the real situation.

```python
def Entropy(img):
    x, y = img.shape
    temp = np.zeros((1,256))

    for i in range(x):
    for j in range(y)
    if img[i,j] == 0:
    k = 1
    else:
    k = img(i,j)
    temp[0,k] = temp[0,k] + 1
    temp = temp / (x * y)

    D = 0
    for i in range(1,256):
    if temp[0,i] != 0:
    D = D - temp[0,i] * log(temp[0,i],2)

    return D
```



## 10.  JPEG: No-Reference Perceptual Quality Assessment of JPEG Compressed Images

Use f (x, y) to represent a picture, the size of the picture is MxN, calculate the signal across each horizontal line:
$$
d_h(x,y)=f(x,y+1)-f(x,y), y\ in\ [1,N-1]
$$
Compute the blockiness, which is estimated as the average differences across block boundaries:
$$
B_h=\frac{1}{M([N/8]-1)}\sum_{i=1}^{M}\sum_{j=1}^{[N/8]-1}{|d_h(i,8j)|}
$$
Compute the average absolute difference between in-block image samples:
$$
A_h=\frac{1}{7}[\frac{8}{M(N-1)}\sum_{i=1}^{M}\sum_{j=1}^{N-1}{|d_h(i,j)|-B_h}]
$$
Then compute the zero-crossing (ZC) rate. Zero-crossing means that the product of dh values of adjacent pixels is negative.
$$
{z_h} = \left{ {\begin{array}{*{20}{c}}
1,horizontal\ ZC\ at\ d_h(m,n)\\
0,otherwise
\end{array}} \right.
$$
The horizontal ZC rate then can be estimated as:
$$
Z_h=\frac{1}{M(N-2)}\sum_{i=1}^{M}\sum_{j=1}^{N-2}z_h(m,n)
$$
Similarly calculate several measures in the vertical direction, and finally get the average values of these measures n the horizontal and vertical directions.
$$
B=\frac{B_h+B_v}{2},A=\frac{A_h+A_v}{2},Z=\frac{Z_h+Z_v}{2}
$$
Finally, the quality of the image is defined as:
$$
S=\alpha+\beta B^{\gamma_1}A^{\gamma_2}Z^{\gamma_3}
$$
where 
$$
\alpha=-245.9,\beta=261.9,\gamma_1=-0.0240,\gamma_2=0.0160,\gamma_3=0.0064
$$


## 11. JPEG2：No-Reference Image Quality Assessment  for JPEG/JPEG2000 Coding

Based on 10, the authors redefined the image quality function:
$$
SS=\frac{4}{1+exp(-1.0217(S-3))}+1
$$
 

## 

