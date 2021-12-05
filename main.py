import cv2
import numpy as np
import face_recognition
#Step 1
#load image ,main image with which all others images will be compared
#convert to RBG
#Step2 Finding the faces in image
#Step3 Finding the Encodings of face
#step4 Comparing it -->result true/false

#step1
imgJeff=face_recognition.load_image_file('MinorPImages/AmazonCEOJeff.jpg') #Loads the image --> #Loaded in BGR format
imgJeff=cv2.cvtColor(imgJeff,cv2.COLOR_BGR2RGB) #convert BGR image to RGB

#Step2
faceLocJeffArray=face_recognition.face_locations(imgJeff) #face_locations is now an array listing the co-ordinates of all the people's face present in image!
#Although in this case the image is having only one face, but as it's a array we need to extract the coordinates

faceLocJeff=faceLocJeffArray[0]
JeffEncodings=face_recognition.face_encodings(imgJeff) #it actually gives 128 measurments of face
JeffOneEncodeOfOneImage=JeffEncodings[0]
print("JeffEncodings")
print(JeffEncodings)

#showing a rectangle over face location
top, right, bottom, left = faceLocJeff
start_point=(top,left)
end_point=(bottom,right)
color=(255,0,255)
thickness=2
cv2.rectangle(imgJeff, start_point, end_point, color, thickness)

imgTestJeff=face_recognition.load_image_file('MinorPImages/SideFaceAmazonCEOimage.jpg')
imgTestJeff=cv2.cvtColor(imgTestJeff,cv2.COLOR_BGR2RGB)
faceLocJeffTestArray=face_recognition.face_locations(imgTestJeff)

JeffTestEncodings=face_recognition.face_encodings(imgTestJeff)
print("JeffTestEncodings")
print(JeffTestEncodings)
JeffTestOneEncodeOfOneImage=JeffTestEncodings[0]


top, right, bottom, left = faceLocJeff
start_point=(top,left)
end_point=(bottom,right)
color=(255,0,255)
thickness=2
print("hello")

cv2.rectangle(imgTestJeff, start_point, end_point, color, thickness)


result=face_recognition.compare_faces([JeffOneEncodeOfOneImage],JeffTestOneEncodeOfOneImage)
print(result)
cv2.imshow('Jeff Bezos',imgJeff)
cv2.imshow('JeffBezos_TestImage',imgTestJeff)
cv2.waitKey(0)
