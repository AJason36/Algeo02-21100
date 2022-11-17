import os
import pickle
import math

import numpy
import numpy.linalg

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

def addMtx(total, add):
    if(len(total) == 0):
        return add
    for i in range(len(total)):
        total[i] += add[i]
    return total

def computeMean():
    global result
    total = []
    cnt = 0
    for key in result:
        total = addMtx(total, result[key])
        cnt += 1
    mean = []
    for val in total:
        mean.append(val / cnt)
    return mean

def diffMatrix(mean):
    global result
    diffMat = [[0.0 for j in range(len(result))] for i in range(len(mean))]

    itr = 0
    for key in result:
        for i in range(len(mean)):
            diffMat[i][itr] = result[key][i] - mean[i]
        itr += 1

    return diffMat

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

def householder(mat):
    # I.S. mat matriks persegi
    assert(len(mat) == len(mat[0]))
    Q = numpy.eye(len(mat))
    matT = transpose(mat)
    for i in range(len(mat) - 1):
        u = matT[i]
        for j in range(i):
            u[j] = 0
        if(mat[0][0] > 0):
            u[i] += normaVektor(u)
        else:
            u[i] -= normaVektor(u)
        norm = normaVektor(u)
        for j in range(len(u)):
            u[j] /= norm
        Hi = multiply(transpose([u]), [u])
        for j in range(len(Hi)):
            for k in range(len(Hi)):
                Hi[j][k] *= -2
            Hi[j][j] += 1
        mat = multiply(Hi, mat)
        matT = transpose(mat)
        if i == 0:
            Q = Hi
        else:
            Q = multiply(Q, Hi)

    return numpy.array(Q), numpy.array(mat)

def cek(mat, x):
    EPS = 1e-10
    m = numpy.copy(mat)
    m = numpy.subtract(m, x * numpy.eye(len(m)))
    if(abs(numpy.linalg.det(m)) < EPS):
        return True
    else:
        return False

# Sumber : https://www.andreinc.net/2021/01/25/computing-eigenvalues-and-eigenvectors-using-qr-decomposition
# NMAX = 20
# MAT_TEST = [[i + 2*j for j in range(NMAX)] for i in range(NMAX)]
def eigenFace(matCov, mat):
    K = 200 # Jumlah eigenvector yang akan diambil
    n = len(matCov)
    mt = numpy.copy(matCov)
    evec = numpy.eye(len(matCov))

    # Loop QR Decomposition to approximate eigenvalues
    for x in range(1000):
        s = mt[n-1][n-1]
        smult = s * numpy.eye(n)
        #Q, R = householder(numpy.subtract(mt, smult))
        Q, R = numpy.linalg.qr(numpy.subtract(mt, smult))
        mt = R @ Q
        mt = numpy.add(mt, smult)
        evec = evec @ Q

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
    
    # Menghitung nilai eigenface untuk seluruh gambar
    # nilai eigenface untuk gambar ke-i ada di kolom ke-i
    eigenFaceList = eigenVector @ mat
    print(len(eigenVector), len(eigenVector[0]))
    return eigenVector, eigenFaceList


def tes(mat):
    w, v = numpy.linalg.eig(numpy.array(mat))
    print(w)
    print(v)

if __name__ == "__main__":
    load(os.path.join(dir_path, 'features.pck'))
    mean = computeMean()
    diffMat = diffMatrix(mean)
    covariance = numpy.array( multiply(transpose(diffMat), diffMat) )
    numpy.set_printoptions(4, suppress=True)
    # covariance = numpy.array([[0, 1], [-2, -3]])
    print(covariance)
    print("Hasil perhitungan sendiri:")
    eigenVector, eigenFaceList = eigenFace(covariance, numpy.array(diffMat))
    # print("Hasil perhitungan numpy:")
    # tes(covariance)
    
