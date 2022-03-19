
# coding: utf-8

# In[19]:


import numpy as np
import cv2
import math
 

# Load Yolo

yolo_model=cv2.dnn.readNetFromDarknet('D://ahmed el sallab/yolo projects/yolov3.cfg','D://ahmed el sallab/yolo projects/yolov3.weights')
#classes=['Person','Car']
classes = []
path="D://ahmed el sallab/yolo projects/coco.names"
with open(path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
   
layer_names = yolo_model.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in yolo_model.getUnconnectedOutLayers()]



video='D://ahmed el sallab/yolo projects/project3/Social-Distance-Detector-master/pedestrians.mp4'
cap =cv2.VideoCapture(video)

count =0
count_pic=0
count_off=0
while cap.isOpened():
    # Capture frame-by-frame
    ret, img = cap.read()

    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    yolo_model.setInput(blob)
    outs = yolo_model.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    font = cv2.FONT_HERSHEY_PLAIN

    count_ppl=0
    l=[]
    lf=[]
 

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label=='person':
                cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
                l=[]
                l.append(x)
                l.append(y)
                lf.append(l)
                count_ppl+=1
                
                
    off=0
    for i in range(len(lf)):
        for j in range(i+1,len(lf)):
            d=math.sqrt( ((lf[j][1]-lf[i][1])**2)+((lf[j][0]-lf[i][0])**2) )
            if d<60:
                img = cv2.line(img, (lf[i][0]+15,lf[i][1]+35), (lf[j][0]+15,lf[j][1]+35), (0,0,255), 2)
                off+=1
                count+=1   
    if count>=5:
        print("FRAME "+str(count_pic)+"    People Count : "+str(count_ppl)+"   RL : "+str(off))
        cv2.imwrite('dataset\\img'+str(count_pic)+'.png',img)  # Saving frames in Main Folder
        count_pic+=1
        if off>1:
            cv2.imwrite('offenders\\img'+str(count_off)+'.png',img) # Saving frames in Offenders Folder
            count_off+=1
           

    if count_ppl>=41 and count>=5:
        a="HIGH ALERT "+str(count_ppl)+"people in your area!"
        print(a)  
   

    if count>=5:
        count=0
        off=0
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)


# In[11]:




