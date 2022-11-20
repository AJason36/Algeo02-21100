import os
import cv2

def writeNewTrainingImage(filepath, img):
    dimension = img.shape

    print(dimension)

    if dimension[0] > dimension[1]:
        newHeight = 256
        newWidth = int((256 / dimension[1]) * dimension[0])
    else:
        newWidth = 256
        newHeight = int((256 / dimension[0]) * dimension[1])

    print(newHeight, newWidth)
    img = cv2.resize(img, (newHeight, newWidth))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    namafile = input("Masukkan nama subjek foto: ")
    namafile = namafile.title().replace(" ", "")

    i = 1
    while namafile + "_" + str(i) + ".jpg" in os.listdir(os.path.join(filepath)):
        i += 1
    
    namafile = namafile + "_" + str(i) + ".jpg"

    cv2.imwrite(os.path.join(filepath, namafile), img)


if __name__ == "__main__":
    currDir = os.path.dirname(os.path.realpath(__file__))
    img = cv2.imread(os.path.join(currDir, "EmmaTest.jpg"))
    writeNewTrainingImage(os.path.join(currDir, "../newtest"), img)
