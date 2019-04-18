# this is to ignore warnings and gpu data
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# imports
# =======================================================================================================================
import keras
import cv2
import os
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# =======================================================================================================================


# =====================================main program starts===============================================================


# for coloring prints
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def tryagain():  #just a function to ask user if user wants to try again
    again = input('Do you want to try again?(Y/N)').lower()
    if again == 'y':
        return True
    elif again == 'n':
        return False
    else:
        return tryagain()

def datashow():  #function to ask the user for showing data
    data_show = input('Data is ready. \nChoose an option:\n1. Show data now\n2. Save data as csv file\n3. Save data as json file\n4. Exit\n\n====>')
    if data_show in ['1', '2', '3', '4']:
        return data_show
    return data_show

print(bcolors.BOLD + bcolors.OKGREEN + "Welcome to our face classifier!!" + bcolors.ENDC)
print('\n\n')

print('Give us just a minute. so that we can load up things.')

from keras.models import model_from_json, load_model

import sys

age_model = keras.models.load_model('models\\age_final.model')   #this will load age model
gender_model = keras.models.load_model('models\\gender_3.model')   #this will load gender model

#this commented code is to load model from json file and .h5 files containing weights only
# get platform was used to get clear screen function
'''def get_platform():
    platforms = {
            'linux1': 'Linux',
            'linux2': 'Linux',
            'darwin': 'OS X',
            'win32': 'Windows'
        }
    if sys.platform not in platforms:
        return sys.platform

    return platforms[sys.platform]

platform = get_platform()
if platform == 'Linux' or platform == "OS X":
    clear = 'clear'
else:
    clear = 'cls'
age_json = open(os.path.join('models', 'age_model.json'), 'r')
loaded_model_json = age_json.read()
age_json.close()
age_model = model_from_json(loaded_model_json)
os.system(clear)
# loading gender structure
gender_json = open(os.path.join('models','gender.json'), 'r')
loaded_model_json = gender_json.read()
gender_json.close()
gender_model = model_from_json(loaded_model_json)
os.system(clear)
gender_model.load_weights(os.path.join('models',"gender_weights_f.h5"))
os.system(clear)
age_model.load_weights(os.path.join('models',"age_weights.h5"))
os.system(clear)
'''
os.system('cls')
print(bcolors.BOLD + bcolors.OKGREEN + "We are ready to go!!!" + bcolors.ENDC)
print('\n')
mega = True  #mega = True implies mega loop is running.
while mega:
    print("Choose any of the options")
    choice = input("1. runtime demo\n2. detect from images\n3. Exit\n")
    if choice == '1':
        import cv2
        import random
        from time import sleep

        cap = cv2.VideoCapture(0)
        cap.set(3, 720)  # WIDTH
        cap.set(4, 800)  # HEIGHT
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1,3,flags=0, minSize= (10, 10))
            # Display the resulting frame
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                crop_img = frame[y:y + h, x:x + w]  #cropping just a face part
                # You may need to convert the color.

                img = cv2.resize(crop_img, (224, 224))   #resizing image for model input
                img = img_to_array(img)
                tensor = np.array(img, dtype="float")
                tensor = tensor.astype('float32') / 255
                tensor = np.reshape(tensor, (1, 224, 224, 3))
                pred = age_model.predict(tensor)   #predicting
                age = np.argmax(pred)
                #age = random.randint(0, 100)
                img = cv2.resize(crop_img, (256, 256))
                img = img_to_array(img)
                tensor = np.array(img, dtype="float")
                tensor = tensor.astype('float32') / 255
                tensor = np.reshape(tensor, (1, 256, 256, 3))
                pred = gender_model.predict(tensor)
                gender = np.argmax(pred)
                if int(float(gender)) == 0:
                    color = (192, 203, 255)
                elif int(float(gender)) == 1:
                    color = (0, 0, 255)
                label = "{},{}".format(age, gender)
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3, cv2.LINE_AA)
                cv2.imshow("Age Gender Demo", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    elif choice == '2':
        path = input('enter a path of directory or image\n===>')
        if os.path.isdir(path):
            files = os.listdir(path)
            if  len(files) > 3:
                print(bcolors.WARNING + 'we have detected more than 3 files in image.' + bcolors.ENDC)
                print(bcolors.BOLD + 'Choose an option.' +bcolors.ENDC)
                def doshow():
                    show = input('1. Only print info with file name\n2. Show all images with detected info\n3. Go back\n\n===>')
                    if show in ['1', '2', '3']:
                        return show
                    else:
                        return doshow()
                do_show = doshow()
                if do_show == '2':
                    for file in files:
                        img = cv2.imread(os.path.join(path, file))
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.1, 3, flags=0, minSize=(10, 10))
                        if len(faces) == 0:
                            print('we are sorry but we have not detected any face in the photo given. Sorry for the inconvenience')
                            if tryagain():
                                break
                            else:
                                mega = False
                                break
                        elif 0 < len(faces):
                            for (x, y, w, h) in faces:
                                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                crop_img = img[y:y + h, x:x + w]
                                # You may need to convert the color.
                                img = cv2.resize(crop_img, (224, 224))
                                img = img_to_array(img)
                                tensor = np.array(img, dtype="float")
                                tensor = tensor.astype('float32') / 255
                                tensor = np.reshape(tensor, (1, 224, 224, 3))
                                pred = age_model.predict(tensor)
                                age = np.argmax(pred)
                                # age = random.randint(0, 100)
                                img = cv2.resize(crop_img, (256, 256))
                                img = img_to_array(img)
                                tensor = np.array(img, dtype="float")
                                tensor = tensor.astype('float32') / 255
                                tensor = np.reshape(tensor, (1, 256, 256, 3))
                                pred = gender_model.predict(tensor)
                                gender = np.argmax(pred)
                                if int(float(gender)) == 1:
                                    gender = 'M'
                                elif int(float(gender)) == 0:
                                    gender = 'F'
                                if int(float(gender)) == 0:
                                    color = (147, 20, 255)
                                elif int(float(gender)) == 1:
                                    color = (0, 0, 255)
                                label = "{},{}".format(age, gender)
                                cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3, cv2.LINE_AA)
                                cv2.imshow("Age Gender Demo", img)
                elif do_show == '1':
                    data_dict = {}
                    for file in files:
                        face_dict = {}
                        img = cv2.imread(os.path.join(path, file))
                        if img.all() == None:
                            print('This file is not an image file. Please double check the file')
                            continue
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.1, 3, flags=0, minSize=(10, 10))
                        if len(faces) == 0:
                            print('we are sorry but we have not detected any face in the photo given. Sorry for the inconvenience')
                            if tryagain():
                                break
                            else:
                                mega = False
                                break
                        elif 0 < len(faces):
                            count = 0
                            for (x, y, w, h) in faces:
                                crop_img = img[y:y + h, x:x + w]
                                img = cv2.resize(crop_img, (224, 224))
                                img = img_to_array(img)
                                tensor = np.array(img, dtype="float")
                                tensor = tensor.astype('float32') / 255
                                tensor = np.reshape(tensor, (1, 224, 224, 3))
                                pred = age_model.predict(tensor)
                                age = np.argmax(pred)
                                img = cv2.resize(crop_img, (256, 256))
                                img = img_to_array(img)
                                tensor = np.array(img, dtype="float")
                                tensor = tensor.astype('float32') / 255
                                tensor = np.reshape(tensor, (1, 256, 256, 3))
                                pred = gender_model.predict(tensor)
                                gender = np.argmax(pred)
                                if int(float(gender)) == 1:
                                    gender = 'M'
                                elif int(float(gender)) == 0:
                                    gender = 'F'
                                face_dict[count] = [age, gender]
                                count +=1
                        data_dict[file] = face_dict
                    exit_data_show = False
                    while not exit_data_show:
                        data_show = datashow()
                        if data_show == '1':
                            import pprint
                            pprint.pprint(data_dict)
                            print('\n\n')
                            break
                        elif data_show == '2':
                            print('Currently we do not support csv saving. Please stay tuned for updates')
                            '''import csv
                            name = input('Enter file name: ')
                            try:
                                with open(name, 'w') as csvfile:
                                    writer = csv.DictWriter(csvfile,fieldnames=['file name', 'face data'])
                                    writer.writeheader()
                                    for data in data_dict:
                                        writer.writerow(data)
                                print('file saved!!')
                            except IOError:
                                print("I/O error")'''
                            print('\n')
                            break
                        elif data_show == '3':
                            import json
                            name = input('Enter file name: ')
                            with open(name, 'w') as fp:
                                json.dump(data_dict, fp)
                            print('file saved!!')
                            break
                        elif data_show == '4':
                            exit_data_show = True
                            break
                        else:
                            break
                elif do_show == '3':
                    break
                else:
                    print('Sorry invalid input!! Exiting...')
                    break
            elif 0 < len(files) < 4:
                for file in files:
                    img = cv2.imread(os.path.join(path, file))
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 3, flags=0, minSize=(10, 10))
                    if len(faces) == 0:
                        print(
                            'we are sorry but we have not detected any face in the photo given. Sorry for the inconvenience')
                        if tryagain():
                            break
                        else:
                            mega = False
                            break
                    elif 0 < len(faces):
                        for (x, y, w, h) in faces:
                            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            crop_img = img[y:y + h, x:x + w]
                            # You may need to convert the color.
                            img = cv2.resize(crop_img, (224, 224))
                            img = img_to_array(img)
                            tensor = np.array(img, dtype="float")
                            tensor = tensor.astype('float32') / 255
                            tensor = np.reshape(tensor, (1, 224, 224, 3))
                            pred = age_model.predict(tensor)
                            age = np.argmax(pred)
                            # age = random.randint(0, 100)
                            img = cv2.resize(crop_img, (256, 256))
                            img = img_to_array(img)
                            tensor = np.array(img, dtype="float")
                            tensor = tensor.astype('float32') / 255
                            tensor = np.reshape(tensor, (1, 256, 256, 3))
                            pred = gender_model.predict(tensor)
                            gender = np.argmax(pred)
                            if int(float(gender)) == 1:
                                gender = 'M'
                            elif int(float(gender)) == 0:
                                gender = 'F'
                            if int(float(gender)) == 0:
                                color = (147, 20, 255)
                            elif int(float(gender)) == 1:
                                color = (0, 0, 255)
                            label = "{},{}".format(age, gender)
                            cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3, cv2.LINE_AA)
                            cv2.imshow(file, img)

        elif os.path.isfile(path):
            img = cv2.imread(path)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 3, flags=0, minSize=(10, 10))
            if len(faces) == 0:
                print(
                    'we are sorry but we have not detected any face in the photo given. Sorry for the inconvenience')
                if tryagain():
                    break
                else:
                    mega = False
                    break
            elif 0 < len(faces):
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    crop_img = img[y:y + h, x:x + w]
                    # You may need to convert the color.
                    img = cv2.resize(crop_img, (224, 224))
                    img = img_to_array(img)
                    tensor = np.array(img, dtype="float")
                    tensor = tensor.astype('float32') / 255
                    tensor = np.reshape(tensor, (1, 224, 224, 3))
                    pred = age_model.predict(tensor)
                    age = np.argmax(pred)
                    # age = random.randint(0, 100)
                    img = cv2.resize(crop_img, (256, 256))
                    img = img_to_array(img)
                    tensor = np.array(img, dtype="float")
                    tensor = tensor.astype('float32') / 255
                    tensor = np.reshape(tensor, (1, 256, 256, 3))
                    pred = gender_model.predict(tensor)
                    gender = np.argmax(pred)
                    if int(float(gender)) == 1:
                        gender = 'M'
                    elif int(float(gender)) == 0:
                        gender = 'F'
                    if gender == 'M':
                        color = (147, 20, 255)
                    elif gender == 'F':
                        color = (0, 0, 255)
                    label = "{},{}".format(age, gender)
                    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3, cv2.LINE_AA)
                    cv2.imshow(path, img)
        else:
            print("Sorry error occurred!! Please cross check the path entered")



    elif choice == '3':
        break
    else:
        continue
