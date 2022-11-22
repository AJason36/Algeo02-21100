import os
import pickle
import math
import time

import numpy
import numpy.linalg

import qr_decomposition

# Directory sekarang
dir_path = os.path.dirname(os.path.realpath(__file__))
result = {}


def takeFirst(elem):
    # Mengambil elemen pertama dari list untuk sorting.
    return elem[0]


def load(dbPath):
    # Load file .pck berisi hasil extract training image
    global result
    with open(dbPath, 'rb') as fp:
        result = pickle.load(fp)


def computeMean():
    # Mengembalikan matriks 1 * n^2 berisi rata-rata training image
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
    # Mengembalikan matriks selisih dengan rata-rata
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


def normalizeVec(vec):
    # Mengembalikan vektor satuan (normalized vector)
    norm = qr_decomposition.normaVektor(vec)

    if norm == 0:
        return vec
    return numpy.divide(vec, norm)


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
        eigenVector[i] = normalizeVec(eigenVector[i])
    
    mat = numpy.transpose(mat)
    
    
    # Menghitung nilai eigenface untuk seluruh gambar
    # nilai eigenface untuk gambar ke-i ada di kolom ke-i
    eigenFaceList = [[0 for j in range(len(eigenVector))] for i in range(len(mat))]
    for i in range(len(mat)):
        for j in range(len(eigenVector)):
            eigenFaceList[i][j] = numpy.dot(mat[i], eigenVector[j])
    return eigenVector, eigenFaceList


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
    

if __name__ == "__main__":
    main()
    