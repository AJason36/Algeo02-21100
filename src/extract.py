# source: https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774
# TODO : tulis source di laporan
import cv2
import numpy
import os
import pickle
# Directory sekarang
dir_path = os.path.dirname(os.path.realpath(__file__))

def extract_features(imgPath, vector_size = 32):
    img = cv2.imread(imgPath)
    try:
        alg = cv2.KAZE_create()
        kps = alg.detect(img)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        kps, dsc = alg.compute(img, kps)
        dsc = dsc.flatten()
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
    maxCnt = 120
    for f in files:
        print('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = extract_features(f)
        maxCnt -= 1
        if maxCnt == 0:
            break
    
    # saving all our feature vectors in pickled file
    with open(pickled_db_path, 'wb') as fp:
        pickle.dump(result, fp)

batch_extractor(os.path.join(dir_path, '../test'), os.path.join(dir_path, '../src/features.pck'))


