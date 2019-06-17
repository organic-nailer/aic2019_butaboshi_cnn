import cv2
import cv2
import numpy as np
from matplotlib import  pyplot as plt

def preprocessing(input_data):
    
   output_size =np.zeros((10,720, 7200,1), np.uint8)

   for data_index in range(10):
       dsts = input_data[data_index,:,:,:,:]
       dsts = dsts.astype(np.uint8)
       dsts_mask = np.zeros((10,720, 720, 3), np.uint8)

       index = 0
       for dst in dsts:
           dst_ismasked = False
           dst = cv2.cvtColor(dst,cv2.COLOR_RGBA2GRAY)
           _, dst = cv2.threshold(dst, 230, 255, cv2.THRESH_BINARY)

           ＿, dst_contours,＿ = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
           dst_contours.sort(key=cv2.contourArea, reverse=True)


           for contour in dst_contours:
               x,y,w,h = cv2.boundingRect(contour)
               arclen = cv2.arcLength(contour,
                                       True) # 対象領域が閉曲線の場合、True
               approx = cv2.approxPolyDP(contour,
                                           0.001*arclen,  # 近似の具合?
                                           True)

               area = cv2.contourArea(approx)


               if area > 45000 and area < 250000:
                   dst_ismasked = True
                   dsts_mask[index] = cv2.rectangle(dsts_mask[index],(x,y),(x+w,y+h),(255,255,255),-1)



           if not dst_ismasked:
               dsts_mask[index,:,:,:]=255


           index+=1

       output = np.concatenate(dsts, axis = 1)
       output_mask = np.concatenate(dsts_mask, axis = 1)
       output = cv2.bitwise_and(output, output_mask)
       output = cv2.cvtColor(output,cv2.COLOR_RGBA2GRAY)

       output_size[data_index,:,:,0]=output


   return output_size



"""
for i in range(20):
   input_data = np.load("intput_data_{0}00.npy".format(i),allow_pickle=True)
   preprocessing(input_data)
"""
input_data = np.load("intput_data_100.npy",allow_pickle=True)
preprocessing(input_data)