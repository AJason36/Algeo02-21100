import numpy
import math

def normaVektor(vec):
    # Mengembalikan norma vektor
    ret = 0.0
    for x in vec:
        ret += x * x

    return math.sqrt(ret)

def householder(matSrc):
    # Mengembalikan Q dan R hasil QR decomposition
    # Menggunakan householder triangularization
    mat = numpy.array(matSrc, dtype='double')
    R = numpy.copy(matSrc)
    Q = numpy.eye(len(matSrc))
    H_i = []
    for i in range(len(mat)):
        a = numpy.copy(mat[:, 0])
        norm_a = normaVektor(a)
        v = a
        if(a[0] > 0):
            v[0] += norm_a
        else:
            v[0] -= norm_a

        if(numpy.dot(v,v) < 1e-8):
            break
        H = numpy.subtract(
            numpy.eye(len(mat)), 
            numpy.multiply(2.0 / numpy.dot(v, v) , numpy.transpose([v]) @ [v])
        )
        H_i += [H]
        
        mat = H @ mat
        R[i, i:] = numpy.copy(mat[0, :])
        R[i:, i] = numpy.copy(mat[:, 0])
        mat = mat[1:, 1:]

    for i in range(len(H_i)-1, -1, -1):
        Q[i:, i:] = H_i[i] @ Q[i:, i:]
    return Q, R


if __name__ == "__main__":
    # MAT_TES = numpy.array([
    #     [2, -2, 18],
    #     [2, 1, 0],
    #     [1, 2, 0]
    # ])
    MAT_TES = numpy.array([
        [1,2,4],
        [0,0,5],
        [0,3,6]
    ])
    Q, R = householder(MAT_TES)
    print(Q)
    print(R)
    print(Q @ R)
