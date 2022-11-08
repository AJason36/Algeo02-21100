import os
import pickle

# Directory sekarang
dir_path = os.path.dirname(os.path.realpath(__file__))
LOADED = False
result = {}

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
    ret = [[0.0 for j in range(len(mat2[0]))] for i in range(len(mat1))]
    if len(mat1[0]) != len(mat2):
        print('Cannot Multiply!')
        return None
    
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                ret[i][j] += mat1[i][k] * mat2[k][j]
    
    return ret

load(os.path.join(dir_path, 'features.pck'))
mean = computeMean()
diffMat = diffMatrix(mean)
covariance = multiply(transpose(diffMat), diffMat)


