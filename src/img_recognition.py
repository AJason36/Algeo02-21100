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
    mean = np.array(eigenData["__mean"])
    eVec = np.array(eigenData["__eigenVector"])

    imageData = np.array(imageData)
    
    diff = np.subtract(imageData, mean)
    ret = []
    for i in range(len(eVec)):
        ret += [np.dot(diff, eVec[i])]

    return ret


def normalizeVec(vec):
    norm = 0

    for i in range(len(vec)):
        norm += vec[i] ** 2
    norm = math.sqrt(norm)

    if norm == 0:
        return vec
    return np.divide(vec, norm)


def euclideanDistance(newEigenFace, comparedEigenFace):
    ret = 0

    # newEigenFace = normalizeVec(newEigenFace)
    # comparedEigenFace = normalizeVec(comparedEigenFace)

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
            # print(lhs, euclideanDistance(eFace, rhs))

    EPS1 = 10000.0
    EPS2 = 1e6
    if minED < EPS1:
        var = 100 * (1-minED)
        return f"Mirip dengan {namaED}\n dengan jarak {round(minED,3)}", True, namaED
    elif minED < EPS2:
        return "Muka tidak dikenali pada dataset", False, namaED
    else:
        return "Gambar tidak dikenali", False, namaED



if __name__ == "__main__":
    print(imgRecognition("../test/training/EmmaWatson_1.jpg"))
    print(imgRecognition("../test/example/Avriltest.webp"))
    print(imgRecognition("../test/example/EmmaTest2.jpg"))
    print(imgRecognition("../test/example/AlvaroTest.jpg"))
