import pickle
import os
import math
import extract
import numpy as np


def loadImage(namaFileGambar):
    # Mengambil Image
    return extract.extractNewImage(namaFileGambar)


def newEigenFace(eigenData, imageData):
    # Mengambil Data
    mean = np.transpose(np.array([eigenData["__mean"]]))
    eVec = np.array(eigenData["__eigenVector"])

    imageData = np.transpose(np.array([imageData]))

    ret = np.subtract(imageData, mean)
    ret = eVec @ ret

    return ret


def normalizeVec(vec):
    norm = np.linalg.norm(vec)

    if norm == 0:
        return vec
    return vec / norm


def euclideanDistance(newEigenFace, comparedEigenFace):
    ret = 0

    # print(newEigenFace[0] - comparedEigenFace[0])
    # print(newEigenFace[0][0])
    # print(comparedEigenFace[0])

    newEigenFace = normalizeVec(np.transpose(newEigenFace)[0])
    comparedEigenFace = normalizeVec(comparedEigenFace)

    for i in range(len(newEigenFace)):
        ret += (newEigenFace[i] - comparedEigenFace[i]) ** 2

    # print(ret)

    return math.sqrt(ret)


def imgRecognition(namaFile):
    # Directory sekarang
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Mengambil Data Eigen Training Image
    with open(os.path.join(dir_path, 'eigenData.pck'), 'rb') as fp:
        eData = pickle.load(fp)

    # Mengambil EigenFace Gambar yang Dibandingkan
    eFace = newEigenFace(eData, loadImage(namaFile))

    # Mencari Gambar dengan Euclidean Distance terkecil
    minED = -1
    namaED = "NULL"

    for lhs, rhs in eData.items():
        if lhs != "__mean" and lhs != "__eigenVector":
            if minED == -1:
                namaED = lhs
                minED = euclideanDistance(eFace, rhs)
            else:
                temp = euclideanDistance(eFace, rhs)

                if temp < minED:
                    namaED = lhs
                    minED = temp
            print(lhs, euclideanDistance(eFace, rhs))

    return namaED


if __name__ == "__main__":
    #print(imgRecognition("../test/Emma Watson60_2012.jpg"))
    print(imgRecognition("../test/Alex Lawther22_84.jpg"))
    print(imgRecognition("EmmaTest2.jpg"))
