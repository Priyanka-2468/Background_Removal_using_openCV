import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(cv2.CAP_PROPS_FPS,60)
segmentor=SelfiSegmentation()
fpsReader=cvzone.FPS()
imgBg=cv2.imread("1.png")
listImg=os.listdir("Images")
print(listImg)
imgList=[]
for imgPath in listImg:
    img=cv2.imread(f'Images/{imgPath}')
    imgList.append(img)
printlen(imgList)

while True:
    success,img=cap.read()
    imgOut=segmentor.removeBG(img,(255,0,0),threshold=0.8)

    imgStacked=cvzone.stackImages([img,imgOut],2,1)
    _, imgStacked = fpsReader.update(imgStacked,color=(0,0,255))

    cv2.imshow("Image",img)
    cv2.imshow("Image Out",imgOut)
    cv2.waitKey(1)