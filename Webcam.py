import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Download model from https://github.com/akanametov/yolo-face?tab=readme-ov-file and put in same directory as this file
model = YOLO("yolov11n-face.pt")

# Create function just in case you need preprocessing
def get_picture(webcam_image, Dmax=608, Dmin=256):
    edited_pic = webcam_image
    return edited_pic

# Open webcam
cap = cv2.VideoCapture(0)
 
# Send webcam image through the model
while True:
    #Get Image
    success,img = cap.read()
    my_pic = get_picture (img)
    
    # Predict with the model
    results = model(my_pic)      

    ## Get bounding box and label
    result = results[0]
    boxes = result.boxes  # ultralytics.engine.results.Boxes
    
    # Index for the number of people identified: important for tracking
    x = 0

    # Loop through each box
    for box in boxes:
        # box.xyxy is a (1, 4) tensor: [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Optional: confidence and class
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{result.names[cls]}: {conf:.2f}"
        label = "Person" + str(x)

        # Draw the rectangle
        cv2.rectangle(my_pic, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # Draw label
        cv2.putText(my_pic, label, (x1, y1 - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(0, 255, 0), thickness=1)
        x = x+1


    #Display image with rectangle
    cv2.imshow("Detected Image", my_pic)

    # Break when "q" pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
