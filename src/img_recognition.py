import pickle
import os
import math
import extract
import numpy as np
from qr_decomposition import normaVektor


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


def euclideanDistance(newEigenFace, comparedEigenFace):
    # Return euclidean distance of two vectors
    ret = 0

    for i in range(len(newEigenFace)):
        ret += (newEigenFace[i] - comparedEigenFace[i]) ** 2

    return math.sqrt(ret)


def imgRecognition(namaFile):
    # Directory sekarang
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Mengambil Data Eigen Training Image
    with open(os.path.join(dir_path, 'eigenData.pck'), 'rb') as fp:
        eData = pickle.load(fp)

    # Mengambil EigenFace Gambar yang Dibandingkan
    img = loadImage(namaFile)
    if len(img) == 0:
        return "Tidak ada muka yang dikenali", False, ""
    eFace = newEigenFace(eData, img)

    # Mencari Gambar dengan Euclidean Distance terkecil
    minED = -1
    namaED = "NULL"
    sim = 0

    for lhs, rhs in eData.items():
        if lhs != "__mean" and lhs != "__eigenVector":
            if minED == -1:
                namaED = lhs
                minED = euclideanDistance(eFace, rhs)
                sim = np.dot(eFace, rhs) / ( normaVektor(eFace) * normaVektor(rhs) )
            else:
                temp = euclideanDistance(eFace, rhs)

                if temp < minED:
                    namaED = lhs
                    minED = temp
                    sim = np.dot(eFace, rhs) / ( normaVektor(eFace) * normaVektor(rhs) )
            # print(lhs, euclideanDistance(eFace, rhs))

    EPS1 = 0.98
    EPS2 = 0.95
    if sim > EPS1:
        return f"Mirip dengan {namaED}\njarak = {round(minED,3)}\nkemiripan = {round(sim*100, 2)}%", True, namaED
    elif sim > EPS2:
        return "Muka tidak dikenali pada dataset", False, namaED
    else:
        return "Gambar tidak dikenali", False, namaED


if __name__ == "__main__":
    print(imgRecognition("../test/training/EmmaWatson_1.jpg"))
    print(imgRecognition("../test/example/Avriltest.webp"))
    print(imgRecognition("../test/example/EmmaTest2.jpg"))
    print(imgRecognition("../test/example/AlvaroTest.jpg"))
    print(imgRecognition("../test/example/PizzaTest.jpg"))

