# source: https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774

import cv2
import numpy
import os
import pickle


def extract_features(imgPath, vector_size = 32):
    imgSize = 256
    imgInit = cv2.imread(imgPath)
    img = cropImage(imgInit)

    # Jika muka tidak dikenali, return list kosong
    if len(img) <= 0:
        return []

    img = cv2.resize(img, (imgSize, imgSize))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        dsc = img.flatten()
        needed_size = 64 * vector_size
        if dsc.size < needed_size:
            dsc = numpy.concatenate([dsc, numpy.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print('Error: ', e)
        return None
    return dsc


def batch_extractor(images_path, pickled_db_path="features.pck"):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

    result = {}
    maxCnt = 1000
    for f in files:
        name = f.split('/')[-1].lower()
        result[name] = extract_features(f)
        if len(result[name]) == 0:
            result.pop(name)
        maxCnt -= 1
        if maxCnt == 0:
            break
    
    # Saving all our feature vectors in pickled file
    with open(pickled_db_path, 'wb') as fp:
        pickle.dump(result, fp)


def extractNewImage(namaFileGambar):
    # Directory sekarang
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return extract_features(os.path.join(dir_path, f'{namaFileGambar}'))


def extract_folder(path):
    # Directory sekarang
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Membuat features.pck
    batch_extractor(path, os.path.join(dir_path, '../src/features.pck'))
    

def cropImage(img):
    # Source : https://www.geeksforgeeks.org/cropping-faces-from-images-using-opencv-python/
    
    # Directory sekarang
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Convert into grayscale
    grayscaleImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(os.path.join(dir_path, 'haarcascade_frontalface_alt2.xml'))
    
    # Detect faces
    faces = face_cascade.detectMultiScale(grayscaleImg, 1.1, 4)
    
    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faces = img[y:y + h, x:x + w]

    return faces


if __name__ == "__main__":
    # Directory sekarang
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Membuat features.pck
    batch_extractor(os.path.join(dir_path, '../test/training'), os.path.join(dir_path, '../src/features.pck'))
