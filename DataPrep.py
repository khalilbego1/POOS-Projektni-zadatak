import PIL
import numpy as np
import cv2 as cv
import os
from  PIL import Image, ImageDraw,ImageChops,ImageEnhance
brojSlika = 100
#------------------------kreiranje i primjenjivanje maski na osnovu anotacija regiona-----------------------------------
def PrimjeniMasku():
    if not os.path.exists('DataSet/DatasetOpenCV/MaskeSlike'):
        os.makedirs('DataSet/DatasetOpenCV/MaskeSlike')
    i=1;
    while i<=brojSlika:
        text = open('DataSet/DatasetOpenCV/Regioni/gt_img_'+str(i)+'.txt','r')
        lines= text.readlines()
        image = Image.open('DataSet/DatasetOpenCV/Slike/img_'+str(i)+'.jpg')
        width, height = image.size
        result =Image.new('RGB', (width, height), color = 'black')
        for j in lines:
            p=j.split(',')
            polygon = ((int(p[0]),int(p[1])),(int(p[2]),int(p[3])),(int(p[4]),int(p[5])),(int(p[6]),int(p[7])))
            img = Image.new('RGB', (width, height), color = 'black')
            ImageDraw.Draw(img).polygon((polygon), outline=(0,0,0), fill=(255,255,255))
            imageCpy =image.copy()
            resultTemp = ImageChops.multiply(imageCpy,img)
            result = ImageChops.add(result, resultTemp)

        result.save('DataSet/DatasetOpenCV/MaskeSlike/img_'+str(i)+'.jpg')
        i+=1
#------------------------smanjivanje suma koristeci nl means -----------------------------------------------------------
def SmanjiSum():
    if not os.path.exists('DataSet/DatasetOpenCV/BezSuma'):
        os.makedirs('DataSet/DatasetOpenCV/BezSuma')
    i=1
    while i<=brojSlika:
        img =cv.imread('DataSet/DatasetOpenCV/Slike/img_'+str(i)+'.jpg')
        dst = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)#nl means
        cv.imwrite('DataSet/DatasetOpenCV/BezSuma/img_'+str(i)+'.jpg',dst)
        i+=1
#------------------------Maskiranje neostrina koristeci slike sa smanjenim sumom----------------------------------------
def MaskirajNeostrine(k):
    if not os.path.exists('DataSet/DatasetOpenCV/MaskiraneNeostrine'):
        os.makedirs('DataSet/DatasetOpenCV/MaskiraneNeostrine')
    i=1
    while i<=brojSlika:
        img =cv.imread('DataSet/DatasetOpenCV/BezSuma/img_'+str(i)+'.jpg')
        blur = cv.blur(img,(5,5))
        dst = img+k*(img-blur)
        cv.imwrite('DataSet/DatasetOpenCV/MaskiraneNeostrine/img_'+str(i)+'.jpg',dst)
        i+=1
#------------------------filteri za poboljsanje slike-------------------------------------------------------------------
def PoboljsajKontrast(faktor):
    if not os.path.exists('DataSet/DatasetOpenCV/PoboljsaniKontrast'):
        os.makedirs('DataSet/DatasetOpenCV/PoboljsaniKontrast')
    i=1
    while i<=brojSlika:
        img = Image.open('DataSet/DatasetOpenCV/Slike/img_'+str(i)+'.jpg')
        enhancer =ImageEnhance.Contrast(img)
        c_img=enhancer.enhance(faktor)
        c_img.save('DataSet/DatasetOpenCV/PoboljsaniKontrast/img_'+str(i)+'.jpg')
        i+=1
def PoboljsajSvjetlost(faktor):
    if not os.path.exists('DataSet/DatasetOpenCV/PoboljsanaSvjetlost'):
        os.makedirs('DataSet/DatasetOpenCV/PoboljsanaSvjetlost')
    i=1
    while i<=brojSlika:
        img = Image.open('DataSet/DatasetOpenCV/Slike/img_'+str(i)+'.jpg')
        enhancer =ImageEnhance.Brightness(img)
        c_img=enhancer.enhance(faktor)
        c_img.save('DataSet/DatasetOpenCV/PoboljsanaSvjetlost/img_'+str(i)+'.jpg')
        i+=1
def UjednaciHistogram():
    if not os.path.exists('DataSet/DatasetOpenCV/UjednacenHistogram'):
        os.makedirs('DataSet/DatasetOpenCV/UjednacenHistogram')
    i=1
    while i<=brojSlika:
        img =cv.imread('DataSet/DatasetOpenCV/Slike/img_'+str(i)+'.jpg',0)# za histogram moraju biti grayscale slike
        dst = cv.equalizeHist(img)
        cv.imwrite('DataSet/DatasetOpenCV/UjednacenHistogram/img_'+str(i)+'.jpg',dst)
        i+=1
#------------------------kreiranje i popunjavanje foldera test i train--------------------------------------------------
def TestTrain():
        if not os.path.exists('DataSet/DatasetOpenCV/Train'):
            os.makedirs('DataSet/DatasetOpenCV/Train')
        if not os.path.exists('DataSet/DatasetOpenCV/Test'):
            os.makedirs('DataSet/DatasetOpenCV/Test')
        i=1
        while i<=brojSlika:
            img = Image.open('DataSet/DatasetOpenCV/Slike/img_'+str(i)+'.jpg')
            if i%2==0:
                img.save('DataSet/DatasetOpenCV/Test/img_'+str(i)+'.jpg')
            else:
                img.save('DataSet/DatasetOpenCV/Train/img_'+str(i)+'.jpg')
            i+=1

PrimjeniMasku()
SmanjiSum()
MaskirajNeostrine(0.1)
PoboljsajKontrast(1.5)
PoboljsajSvjetlost(1.5)
UjednaciHistogram()
TestTrain()
