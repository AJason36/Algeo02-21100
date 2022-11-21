import os
import pickle
import math
import time

import numpy
import numpy.linalg

import img_recognition
import qr_decomposition

# Directory sekarang
dir_path = os.path.dirname(os.path.realpath(__file__))
LOADED = False
result = {}

def takeFirst(elem):
    return elem[0]

def load(dbPath):
    global LOADED, result
    if not LOADED:
        with open(dbPath, 'rb') as fp:
            result = pickle.load(fp)
        LOADED = True

def computeMean():
    global result
    total = []
    cnt = 0
    for key, val in result.items():
        if cnt == 0:
            total = val
        else:
            total = numpy.add(total, val)
        cnt += 1
    mean = []
    for val in total:
        mean.append(val / cnt)
    return mean

def diffMatrix(mean):
    global result
    diffMat = [[0.0 for j in range(len(result))] for i in range(len(mean))]
    names = []

    itr = 0
    for key in result:
        for i in range(len(mean)):
            diffMat[i][itr] = result[key][i] - mean[i]
        names += [key]
        itr += 1

    return diffMat, names

def transpose(mat):
    matTranspose = [[mat[j][i] for j in range(len(mat))] for i in range(len(mat[0]))]
    return matTranspose

def multiply(mat1, mat2):
    assert(len(mat1[0]) == len(mat2))
    ret = [[0.0 for j in range(len(mat2[0]))] for i in range(len(mat1))]
    if len(mat1[0]) != len(mat2):
        print('Cannot Multiply!')
        return None
    
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                ret[i][j] += mat1[i][k] * mat2[k][j]
    
    return ret

def normaVektor(vec):
    ret = 0.0
    for x in vec:
        ret += x * x
    return math.sqrt(ret)

# Sumber : https://www.andreinc.net/2021/01/25/computing-eigenvalues-and-eigenvectors-using-qr-decomposition
# Menghitung eigen face
# Mengembalikan eigenVector dan eigenFaceList
def eigenFace(matCov, mat, useBuiltIn = False):
    K = 20 # Jumlah eigenvector yang akan diambil
    n = len(matCov)
    mt = numpy.copy(matCov)
    evec = numpy.eye(len(matCov))

    # Loop QR Decomposition to approximate eigenvalues
    ones = numpy.eye(n)
    for x in range(1000):
        s = mt[n-1][n-1]
        smult = s * ones
        if useBuiltIn:
            Q, R = numpy.linalg.qr(numpy.subtract(mt, smult))
        else:
            Q, R = qr_decomposition.householder(numpy.subtract(mt, smult))
        mt = R @ Q
        mt = numpy.add(mt, smult)
        evec = evec @ Q

    # Transpose eigenvector sementara supaya tiap baris menjadi satu vector
    evec = numpy.transpose(evec)
    # print('evec dimension = ', len(evec), 'x', len(evec[0]))

    # Mencari K eigenvectors yang bersesuaian dengan K eigenvalues terbesar
    eigenPairs = []
    for i in range(len(mt)):
        eigenPairs += [[mt[i][i]]]
        for j in range(len(evec[i])):
            eigenPairs[i] += [evec[i][j]]
    eigenPairs.sort(key=takeFirst, reverse=True)

    eigenVector = []
    for i in range(min(K, len(eigenPairs))):
        eigenVector += [eigenPairs[i][1:]]
    for i in range(len(eigenVector)):
        eigenVector[i] = mat @ eigenVector[i]
        eigenVector[i] = img_recognition.normalizeVec(eigenVector[i])
    
    mat = numpy.transpose(mat)
    
    
    # Menghitung nilai eigenface untuk seluruh gambar
    # nilai eigenface untuk gambar ke-i ada di kolom ke-i
    eigenFaceList = [[0 for j in range(len(eigenVector))] for i in range(len(mat))]
    for i in range(len(mat)):
        for j in range(len(eigenVector)):
            eigenFaceList[i][j] = numpy.dot(mat[i], eigenVector[j])
    return eigenVector, eigenFaceList

# Pengecekan dengan library bawaan
def tes(mat):
    w, v = numpy.linalg.eig(numpy.array(mat))
    print(w)
    print(v)

def main():
    numpy.set_printoptions(4, suppress=True)

    # Load file berisi hasil extract gambar
    print('Loading features.pck...')
    start_time = time.time()
    load(os.path.join(dir_path, 'features.pck'))
    end_time = time.time()
    print(f"features.pck loaded successfully in {round(end_time - start_time, 2)} seconds!")

    # Menghitung matrix covariance
    print('Computing difference matrix...')
    mean = computeMean()
    diffMat, names = diffMatrix(mean)
    print('Finished computing difference matrix')
    print('Computing covariance...')
    start_time = time.time()
    covariance = numpy.transpose(diffMat) @ numpy.array(diffMat)
    end_time = time.time()
    print(f'Finished computing covariance in {round(end_time - start_time, 2)} seconds')

    # Menghitung eigenFace
    print('Computing eigenface...')
    start_time = time.time()
    eigenVector, eigenFaceList = eigenFace(covariance, numpy.array(diffMat))
    end_time = time.time()
    print(f'Finished eigenface in {round(end_time-start_time,2)} seconds')

    # Memasukkan hasil perhitungan ke eigenData.pck
    eigenData = {}
    eigenData['__eigenVector'] = eigenVector
    eigenData['__mean'] = mean
    for i in range(len(eigenFaceList)):
        eigenData[names[i]] = eigenFaceList[i]
    with open(os.path.join(dir_path, 'eigenData.pck'), 'wb') as fp:
        pickle.dump(eigenData, fp)

    # Tes dengan library bawaan
    # print("Hasil perhitungan numpy:")
    # tes(covariance)
    

if __name__ == "__main__":
    main()
    