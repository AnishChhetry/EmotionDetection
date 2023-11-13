import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

img= cv2.imread('happy.jpg')
plt.imshow(img) #BGR default
plt.show()
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) #RGB
plt.show()

predictions = DeepFace.analyze(img)
print("All predictions:")
print(predictions)
print("Dominant emotion prediction:")
print(predictions[0]['dominant_emotion'])

faceCascade= cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

faces = faceCascade.detectMultiScale(gray,1.1,4)

#draw a rectangle around the face
for(x,y,w,h) in faces:
  cv2.rectangle(img, (x, y), (x+w,y+h), (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) #RGB
plt.show()

font = cv2.FONT_HERSHEY_SIMPLEX
# use putText() method for inserting text on video

cv2.putText(img,
            predictions[0]['dominant_emotion'],
            (50,50),
            font,3,
            (0, 0, 0),
            2,
            cv2.LINE_4);

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) #RGB
plt.show()

"yoga pose"
