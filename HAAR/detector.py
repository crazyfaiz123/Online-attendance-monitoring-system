# #Copyright Anirban Kar (anirbankar21@gmail.com)
# #
# #   Licensed under the Apache License, Version 2.0 (the "License");
# #   you may not use this file except in compliance with the License.
# #  You may obtain a copy of the License at
# #
# #       http://www.apache.org/licenses/LICENSE-2.0
# #
# #   Unless required by applicable law or agreed to in writing, software
# #   distributed under the License is distributed on an "AS IS" BASIS,
# #   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# #   See the License for the specific language governing permissions and
# #   limitations under the License.
#
# import cv2,os
# import numpy as np
# from PIL import Image
#
# path = os.path.dirname(os.path.abspath(__file__))
#
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read('C:/Users/Farhan/Desktop/New folder/Face-Recognition-master/trainer/trainer.yml')
# cascadePath = path+"/Classifiers/face.xml"
# faceCascade = cv2.CascadeClassifier('C:/Users/Farhan/Desktop/New folder/Face-Recognition-master/haarcascade_frontalface_default.xml')
#
# # cam = cv2.VideoCapture(0)
# font = cv2.FONT_HERSHEY_SIMPLEX #Creates a font
#
# im=cv2.imread('C:/Users/Farhan/Desktop/Images/Farhan.JPG')
# gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=10, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
# for(x,y,w,h) in faces:
#         nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
#         cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
#         if(nbr_predicted==1):
#              nbr_predicted='Abhijeet'
#         elif(nbr_predicted==2):
#              nbr_predicted='Abhishek'
#         elif (nbr_predicted == 3):
#             nbr_predicted = 'Aditya'
#         elif (nbr_predicted == 4):
#             nbr_predicted = 'Adi'
#         elif (nbr_predicted == 5):
#             nbr_predicted = 'Akash'
#         elif (nbr_predicted == 6):
#             nbr_predicted = 'Anand'
#         elif (nbr_predicted == 7):
#             nbr_predicted = 'Ankit'
#         elif (nbr_predicted == 8):
#             nbr_predicted = 'Anshuman'
#         elif (nbr_predicted == 9):
#             nbr_predicted = 'Anubhav'
#         elif (nbr_predicted == 10):
#             nbr_predicted = 'Anuj'
#         elif (nbr_predicted == 11):
#             nbr_predicted = 'Arya'
#         elif (nbr_predicted == 12):
#             nbr_predicted = 'Ashwani'
#         elif (nbr_predicted == 13):
#             nbr_predicted = 'Avinash'
#         elif (nbr_predicted == 14):
#             nbr_predicted = 'Ayushamn'
#         elif (nbr_predicted == 15):
#             nbr_predicted = 'Bhardwaj'
#         elif (nbr_predicted == 16):
#             nbr_predicted = 'Deepak'
#         elif (nbr_predicted == 17):
#             nbr_predicted = 'Dileep'
#         elif (nbr_predicted == 18):
#             nbr_predicted = 'Diwakar'
#         elif (nbr_predicted == 19):
#             nbr_predicted = 'Durgesh'
#         elif (nbr_predicted == 20):
#             nbr_predicted = 'Gulab'
#         elif (nbr_predicted == 21):
#             nbr_predicted = 'Jayanti'
#         elif (nbr_predicted == 22):
#             nbr_predicted = 'Jitendra'
#         elif (nbr_predicted == 23):
#             nbr_predicted = 'Kaushal'
#         elif (nbr_predicted == 24):
#             nbr_predicted = 'Keerti'
#         elif (nbr_predicted == 25):
#             nbr_predicted = 'Mani'
#         elif (nbr_predicted == 26):
#             nbr_predicted = 'Nanci'
#         elif (nbr_predicted == 27):
#             nbr_predicted = 'Nitesh'
#         elif (nbr_predicted == 28):
#             nbr_predicted = 'Palak'
#         elif (nbr_predicted == 29):
#             nbr_predicted = 'Pramod'
#         elif (nbr_predicted == 30):
#             nbr_predicted = 'Pranjal'
#         elif (nbr_predicted == 31):
#             nbr_predicted = 'Prashant'
#         elif (nbr_predicted == 32):
#             nbr_predicted = 'Priyanka'
#         elif (nbr_predicted == 33):
#             nbr_predicted = 'Randhir'
#         elif (nbr_predicted == 34):
#             nbr_predicted = 'Ranjana'
#         elif (nbr_predicted == 35):
#             nbr_predicted = 'Rishikesh'
#         elif (nbr_predicted == 36):
#             nbr_predicted = 'Rumi'
#         elif (nbr_predicted == 37):
#             nbr_predicted = 'Sadhna'
#         elif (nbr_predicted == 38):
#             nbr_predicted = 'Sakshi'
#         elif (nbr_predicted == 39):
#             nbr_predicted = 'Satendra'
#         elif (nbr_predicted == 40):
#             nbr_predicted = 'Shahnawaz'
#         elif (nbr_predicted == 41):
#             nbr_predicted = 'Shivam'
#         elif (nbr_predicted == 42):
#             nbr_predicted = 'Shivangi'
#         elif (nbr_predicted == 43):
#             nbr_predicted = 'Shubham'
#         elif (nbr_predicted == 44):
#             nbr_predicted = 'Sidhart'
#         elif (nbr_predicted == 45):
#             nbr_predicted = 'Sonam'
#         elif (nbr_predicted == 46):
#             nbr_predicted = 'Sushma'
#         elif (nbr_predicted == 47):
#             nbr_predicted = 'Ujala'
#         elif (nbr_predicted == 48):
#             nbr_predicted = 'Vishesh'
#         elif (nbr_predicted == 49):
#             nbr_predicted = 'Vishwakarma'
#         elif (nbr_predicted == 50):
#             nbr_predicted = 'Vishwas'
#         elif (nbr_predicted == 51):
#             nbr_predicted = 'Vivek'
#         elif (nbr_predicted == 52):
#             nbr_predicted = 'Yash'
#         elif (nbr_predicted == 53):
#             nbr_predicted = 'Yashvendra'
#         else:
#             nbr_predicted = 'Unknown'
#
#
#
#
#         cv2.putText(im,str(nbr_predicted)+"--"+str(conf), (x,y+h),font, 1.1, (0,255,0)) #Draw the text
#
#         cv2.imshow('im',im)
#         cv2.waitKey(0)
#
#
#
#
import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('C:/Users/Farhan/Desktop/New folder/Face-Recognition-master/trainer/trainer.yml')

cascadePath = "C:/Users/Farhan/Desktop/New folder/Face-Recognition-master/haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX


im=cv2.imread('C:/Users/Farhan/Desktop/IMG_8177.jpg')
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray, 1.2,5)

for(x,y,w,h) in faces:

        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])


        print(conf,"\t",nbr_predicted)

        if (nbr_predicted == 1):
            nbr_predicted = 'Abhijeet'
        elif (nbr_predicted == 2):
            nbr_predicted = 'Abhishek'
        elif (nbr_predicted == 3):
            nbr_predicted = 'Aditya'
        elif (nbr_predicted == 4):
            nbr_predicted = 'Adi'
        elif (nbr_predicted == 5):
            nbr_predicted = 'Akash'
        elif (nbr_predicted == 6):
            nbr_predicted = 'Anand'
        elif (nbr_predicted == 7):
            nbr_predicted = 'Ankit'
        elif (nbr_predicted == 8):
            nbr_predicted = 'Anshuman'
        elif (nbr_predicted == 9):
            nbr_predicted = 'Anubhav'
        elif (nbr_predicted == 10):
            nbr_predicted = 'Anuj'
        elif (nbr_predicted == 11):
            nbr_predicted = 'Arya'
        elif (nbr_predicted == 12):
            nbr_predicted = 'Ashwani'
        elif (nbr_predicted == 13):
            nbr_predicted = 'Avinash'
        elif (nbr_predicted == 14):
            nbr_predicted = 'Ayushamn'
        elif (nbr_predicted == 15):
            nbr_predicted = 'Bhardwaj'
        elif (nbr_predicted == 16):
            nbr_predicted = 'Deepak'
        elif (nbr_predicted == 17):
            nbr_predicted = 'Dileep'
        elif (nbr_predicted == 18):
            nbr_predicted = 'Diwakar'
        elif (nbr_predicted == 19):
            nbr_predicted = 'Durgesh'
        elif (nbr_predicted == 20):
            nbr_predicted = 'Gulab'
        elif (nbr_predicted == 21):
            nbr_predicted = 'Jayanti'
        elif (nbr_predicted == 22):
            nbr_predicted = 'Jitendra'
        elif (nbr_predicted == 23):
            nbr_predicted = 'Kaushal'
        elif (nbr_predicted == 24):
            nbr_predicted = 'Keerti'
        elif (nbr_predicted == 25):
            nbr_predicted = 'Mani'
        elif (nbr_predicted == 26):
            nbr_predicted = 'Nanci'
        elif (nbr_predicted == 27):
            nbr_predicted = 'Nitesh'
        elif (nbr_predicted == 28):
            nbr_predicted = 'Palak'
        elif (nbr_predicted == 29):
            nbr_predicted = 'Pramod'
        elif (nbr_predicted == 30):
            nbr_predicted = 'Pranjal'
        elif (nbr_predicted == 31):
            nbr_predicted = 'Prashant'
        elif (nbr_predicted == 32):
            nbr_predicted = 'Priyanka'
        elif (nbr_predicted == 33):
            nbr_predicted = 'Randhir'
        elif (nbr_predicted == 34):
            nbr_predicted = 'Ranjana'
        elif (nbr_predicted == 35):
            nbr_predicted = 'Rishikesh'
        elif (nbr_predicted == 36):
            nbr_predicted = 'Rumi'
        elif (nbr_predicted == 37):
            nbr_predicted = 'Sadhna'
        elif (nbr_predicted == 38):
            nbr_predicted = 'Sakshi'
        elif (nbr_predicted == 39):
            nbr_predicted = 'Satendra'
        elif (nbr_predicted == 40):
            nbr_predicted = 'Shahnawaz'
        elif (nbr_predicted == 41):
            nbr_predicted = 'Shivam'
        elif (nbr_predicted == 42):
            nbr_predicted = 'Shivangi'
        elif (nbr_predicted == 43):
            nbr_predicted = 'Shubham'
        elif (nbr_predicted == 44):
            nbr_predicted = 'Sidhart'
        elif (nbr_predicted == 45):
            nbr_predicted = 'Sonam'
        elif (nbr_predicted == 46):
            nbr_predicted = 'Sushma'
        elif (nbr_predicted == 47):
            nbr_predicted = 'Ujala'
        elif (nbr_predicted == 48):
            nbr_predicted = 'Vishesh'
        elif (nbr_predicted == 49):
            nbr_predicted = 'Vishwakarma'
        elif (nbr_predicted == 50):
            nbr_predicted = 'Vishwas'
        elif (nbr_predicted == 51):
            nbr_predicted = 'Vivek'
        elif (nbr_predicted == 52):
            nbr_predicted = 'Yash'
        elif (nbr_predicted == 53):
            nbr_predicted = 'Yashvendra'
        else:
            nbr_predicted = 'Unknown'


        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(nbr_predicted), (x,y-40), font, 2, (255,255,255), 3)
cv2.namedWindow('im', cv2.WINDOW_NORMAL)
cv2.imshow('im',im)



cv2.waitKey(0)
#
#
#
#
