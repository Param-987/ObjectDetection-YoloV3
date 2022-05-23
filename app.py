import cv2
import numpy as np
import matplotlib.pyplot as plt

def hrgb(value):
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

yolo = cv2.dnn.readNet("./yolov3.weights","./yolov3.cfg")
classes = []
with open("./yolov3.txt",'r') as f:
    classes = f.read().split('\n')

image = cv2.imread('./dog.jpeg')

(height,width) = image.shape[:2]

blob = cv2.dnn.blobFromImage(image,1/255,(480,480),(0,0,0),swapRB=True,crop=False)

yolo.setInput(blob)
layer_outputs = yolo.forward(yolo.getUnconnectedOutLayersNames())

boxes = []
confidences = []
class_ids = []
conf_threshold = 0.5
nms_threshold = 0.4
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
        if confidence > 0.7:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            
            x = center_x-w/2
            y = center_y-h/2
                # update our list of bounding box coordinates,
                # confidences, and class IDs
            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

Font = cv2.FONT_HERSHEY_COMPLEX_SMALL
Color = np.random.uniform(0,255,size=(len(boxes),3))

for i in indices.flatten():
    x,y,w,h = boxes[i]
    label = str(classes[class_ids[i]])
    confi = str(round(confidences[i],2))
    color = Color[i]
    x,y = int(x),int(y)
    
    cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
    cv2.putText(image,label+" "+confi,(x+5,y+20),Font,1,(0,0,0),1)

plt.figure(figsize=(8,8))
cv2.imshow("image", image)
cv2.waitKey()

# cv2.imwrite("object-detection.jpg", image)
# cv2.destroyAllWindows()