import cv2
import time
import os
from extract import cropImage


def videoCam():
    # Directory sekarang
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Menginisialisasi Video
    cap = cv2.VideoCapture(0)
    # scaling_factor = 1.5
    start = time.time()

    # Menginisialisasi Arr of Captured Frame
    arr = []
    
    # Loop Video sampai ESC diklik
    while True:
        # Capture Frame saat ini
        ret, frame = cap.read()
        
        # Meresize Frame
        # frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    
        # Menampilkan isi Frame
        cv2.imshow('Webcam', frame)

        # Mendeteksi apakah ESC diklik
        c = cv2.waitKey(1)
        if c == 27:
            break

        # Mengecek apakah sudah 15 detik
        end = time.time()

        if (int(end - start) == 15):
            frame = cropImage(frame)
            cv2.imwrite(os.path.join(dir_path, "Hehe.png"), frame)
            arr += [frame]
            print("Photo Taken")
            start = time.time()

    # Menutup dan Memberhentikan Video Capture
    cap.release()
    cv2.destroyAllWindows()
    
    # Meminta apakah mau menyimpan foto ke dataset
    isSave = input("Save foto yang telah diambil ke dataset? (Y/N): ")

    # Mensave foto ke dataset apabila input Y
    if (isSave == 'Y'):
        # Meminta nama subjek
        namaSubjek = input("Masukkan nama subjek: ")
        namaSubjek = namaSubjek.title().replace(" ", "")

        # Looping write image untuk setiap foto
        i = 1
        for photo in arr:
            temp = namaSubjek
            while temp + "_" + str(i) + ".jpg" in os.listdir(os.path.join(dir_path, "../test/training/")):
                i += 1
            
            temp = temp + "_" + str(i) + ".jpg"
            print(temp)
            temp = "../test/training/" + temp

            cv2.imwrite(os.path.join(dir_path, temp), photo)


if __name__ == "__main__":
    videoCam()
