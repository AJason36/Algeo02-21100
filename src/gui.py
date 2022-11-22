import os
import tkinter as tk
import tkinter.messagebox
import customtkinter as ctk
from tkinter import ttk
from tkinter import *
from tkinter import filedialog as fd
from tkinter.ttk import *
from PIL import Image, ImageTk
from threading import Thread
import eigenface
import img_recognition
import extract
from pathlib import Path 
import time 
from extract import cropImage
import cv2

ctk.set_appearance_mode("System")  # Modes: system (default), light, dark
ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Face Algorithm M")
        self.geometry("1000x600")
        self.protocol("WM_DELETE_WINDOW",self.on_close)
        self.folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../test/training")
        # configure row column
        self.columnconfigure(0,weight=1)
        self.rowconfigure(0,weight=1)
        self.rowconfigure(1,weight=6)
        self.rowconfigure(2,weight=3)

        # setup frame
        self.frame_top = ctk.CTkFrame(self, width=600,height=400, corner_radius=0, fg_color='black')
        self.frame_mid = ctk.CTkFrame(self, width=600,height=400, corner_radius=0, fg_color='#3B3B3B')
        self.frame_bot = ctk.CTkFrame(self, width=600,height=400, corner_radius=0, fg_color='#545454')

        self.frame_top.grid(row=0,column=0, sticky='nsew')
        self.frame_mid.grid(row=1,column=0, sticky='nsew')
        self.frame_bot.grid(row=2,column=0, sticky='nsew')

        self.frame_top.columnconfigure(0,weight=1)
        # self.frame_top.columnconfigure(1,weight=1)
        # self.frame_top.columnconfigure(2,weight=1)

        self.frame_mid.columnconfigure(0, weight=4)
        self.frame_mid.columnconfigure(1, weight=1)
        self.frame_mid.columnconfigure(2, weight=4)
        self.frame_mid.rowconfigure(0, weight=1)
        self.frame_mid.rowconfigure(1, weight=1)
        self.frame_mid.rowconfigure(2, weight=1)
        self.frame_mid.rowconfigure(3, weight=1)

        self.frame_bot.rowconfigure(0, weight=1)
        self.frame_bot.rowconfigure(1,weight=1)

        self.frame_input = ctk.CTkFrame(master = self.frame_mid, width=200, height=300, corner_radius=10)
        self.frame_res =  ctk.CTkFrame(master = self.frame_mid,  width=200, height=300, corner_radius=10)
        self.frame_button = ctk.CTkFrame(master = self.frame_mid, width=20, height=10, corner_radius=0, fg_color='#3B3B3B')
        self.frame_title_test = ctk.CTkFrame(master = self.frame_mid, width=20, height=10, corner_radius=0, fg_color='#3B3B3B')
        self.frame_button_res = ctk.CTkFrame(master = self.frame_mid, width=20, height=10, corner_radius=0, fg_color='#3B3B3B')
        self.frame_title_res = ctk.CTkFrame(master = self.frame_mid, width=20, height=10, corner_radius=0, fg_color='#3B3B3B')

        self.frame_execution = ctk.CTkFrame(master = self.frame_bot, width=20, height=10, corner_radius=0, fg_color='#545454')
        self.frame_time = ctk.CTkFrame(master = self.frame_bot, width=20, height=10, corner_radius=0, fg_color='#545454')

        self.frame_input.grid(row=0,column=0, sticky='nsew', padx=10, pady=(2,0))
        self.frame_button.grid(row=1,column=0, sticky='nsew', padx=10, pady=0)
        self.frame_title_test.grid(row=2,column=0, sticky='nsew', padx=10, pady=0)
        self.frame_res.grid(row=0,column=2, sticky='nsew', padx=10, pady=(2,0))
        self.frame_button_res.grid(row=1,column=2, sticky='nsew', padx=10, pady=0)
        self.frame_title_res.grid(row=2,column=2, sticky='nsew', padx=10, pady=0)
        
        self.frame_execution.grid(row=0,column=0, sticky='nsew', padx=10, pady=(10,0))
        self.frame_time.grid(row=1,column=0, sticky='nsew', padx=20, pady=0)
        
        self.frame_button.columnconfigure(0,weight=1)
        self.frame_button.columnconfigure(1,weight=1)

        self.frame_button_test = ctk.CTkFrame(master = self.frame_button, width=20, height=10, corner_radius=0, fg_color='#3B3B3B')
        self.frame_button_cam = ctk.CTkFrame(master = self.frame_button, width=20, height=10, corner_radius=0, fg_color='#3B3B3B')
        self.frame_button_test.grid(row=1,column=0, sticky='nsew', padx=10, pady=0)
        self.frame_button_cam.grid(row=1,column=1, sticky='nsew', padx=10, pady=0)
        # print image kosong
        self.insert_button = ctk.CTkButton(master=self, text = 'Insert Image',width=20,command=self.insert_img)

        self.file1 = 'empty.png'
        self.folder1 = 'img'
        # Open dan show image
        self.dir1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join(self.folder1, self.file1))
        self.img = Image.open(self.dir1)
        self.img_resized = self.img.resize((400,300))
        self.imgtk1 = ImageTk.PhotoImage(self.img_resized)
        self.imgtk2= ImageTk.PhotoImage(self.img_resized)
        # setup label
        self.label_top = ctk.CTkLabel(master = self.frame_top, text='FACE RECOGNITION ALGORITHM',text_font = ('Verdana',20,'bold'), justify=CENTER)
        self.label_input = ctk.CTkLabel(master = self.frame_mid, image = self.imgtk1)
        self.label_res = ctk.CTkLabel(master = self.frame_mid,image = self.imgtk2)
        self.label_tes = ctk.CTkLabel(master = self.frame_mid,text='Test Image', text_font = ("Verdana",14, 'bold'))
        self.label_text_tes = ctk.CTkLabel(master = self.frame_mid,text='No Image Selected', text_font = ("Verdana",12))
        self.label_tres = ctk.CTkLabel(master = self.frame_mid,text='Closest Result',text_font = ("Verdana",14,'bold'))
        self.label_text_res = ctk.CTkLabel(master = self.frame_mid,text='No Folder Selected', text_font = ("Verdana",12))
        self.label_bot = ctk.CTkLabel(master = self.frame_execution, text='Execute Time: ',text_font = ("Verdana",12,'bold'))
        self.label_time = ctk.CTkLabel(master = self.frame_time, text='',text_font = ("Verdana",12,'bold'))
        
        self.label_input.place(relx =0.0,rely=0.0, anchor=tkinter.N)
        self.label_top.place(relx=0.5,rely=0.5,anchor=CENTER)

        self.label_top.grid(row=0,column=1,padx=5,pady=2, sticky='new')
        self.label_input.grid(row=0, column=0,padx=0,pady=2)
        self.label_res.grid(row=0, column=2,padx=0,pady=2)
        self.label_tes.grid(row=2, column=0,padx=0,pady=0)
        self.label_text_tes.grid(row=3, column=0)
        self.label_tres.grid(row=2, column=2,padx=0,pady=0)
        self.label_text_res.grid(row = 3,column=2)

        self.label_bot.grid(row=0, column=0,padx=10,pady=2)
        self.label_top.place(relx=0.5,rely=0.5,anchor=E)
        self.label_time.grid(row=1, column=0,padx=5,pady=2,sticky='w')
        
        # set up button
        self.label_top.grid(row=0,column=0,padx=10,pady=2)
        self.insert_button = ctk.CTkButton(master=self.frame_button, text = 'Insert Image',text_font = ("Cordia",12),width=50,height=40,command=self.insert_img)
        self.insert_button.grid(row=1, column=0,padx=10,pady=5)
        self.insert_button = ctk.CTkButton(master=self.frame_button, text = 'Open Cam',text_font = ("Cordia",12),width=50,height=40,command=self.init_cam)
        self.insert_button.grid(row=1, column=1,padx=10,pady=5)

        self.insert_folder = ctk.CTkButton(master=self.frame_mid, text = 'Insert Folder',text_font = ("Cordia",12),width=100,height=40,command=self.insert_folder)
        self.insert_folder.grid(row=1, column=2,padx=10)

    def insert_img(self):        
        #open file
        self.filename = fd.askopenfilename()
        self.label_text_tes.configure(text = Path(self.filename).name)
        self.dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.filename)
        self.img2 = Image.open(self.dir)
        self.img_resized = self.img2.resize((600,500))
        self.imgtk1 = ImageTk.PhotoImage(self.img_resized)
        self.label_input = ctk.CTkLabel(master = self.frame_mid, image = self.imgtk1)
        self.label_input.grid(row=0, column=0,padx=0,pady=2)
        #processing image input
        start_time = time.time()
        msg, isRecognized, fileName = img_recognition.imgRecognition(self.dir)
        self.label_bot.configure(text =f'Execution Time: {round(time.time()-start_time,2)} second')
        self.label_text_res.configure(text = msg)
        self.label_text_res.configure(text = msg)
        if isRecognized:
            temp = Path(fileName).name
            self.dirTemp =os.path.join(self.folder, temp)
            self.img2 = Image.open(self.dirTemp)
            self.img_resized = self.img2.resize((600,500))
            self.imgtk2 =ImageTk.PhotoImage(self.img_resized)
            self.label_res.configure(image = self.imgtk2)
        else:
            self.file1 = 'unidentified.png'
            self.folder1 = 'img'
            # Open dan show image
            self.dir1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join(self.folder1, self.file1))
            self.img = Image.open(self.dir1)
            self.img_resized = self.img.resize((300,400))
            self.imgtk1 = ImageTk.PhotoImage(self.img_resized)
            self.label_res.configure(image = self.imgtk1)
        
        # self.labelimg=ctk.CTkLabel(master=self,image=self.imgtk)
        # self.labelimg.pack(padx=10,pady=10)
    

    def init_cam(self):
        self.cap = cv2.VideoCapture(0)
        self.startCamTime = time.time()
        self.open_cam()

    def open_cam(self):
        _, frame = self.cap.read()
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        self.img = Image.fromarray(cv2image)
        self.imgtk = ImageTk.PhotoImage(image=self.img)
        self.label_input.configure(image=self.imgtk)
        
        if (int(time.time() - self.startCamTime) == 15):
            start_time = time.time()
            self.dir = os.path.dirname(os.path.realpath(__file__))
            cv2.imwrite(os.path.join(self.dir, "../test/example/test.jpg"), frame)
            self.label_text_tes.configure(text = 'test.jpg')
            msg, isRecognized, fileName = img_recognition.imgRecognition(os.path.join(self.dir, "../test/example/test.jpg"))
            self.label_bot.configure(text =f'Execution Time: {round(time.time() - start_time,2)} second')
            self.label_text_res.configure(text = msg)
            self.label_text_res.configure(text = msg)

            if isRecognized:
                temp = Path(fileName).name
                self.dirTemp = os.path.join(self.folder, temp)
                self.img2 = Image.open(self.dirTemp)
                self.img_resized = self.img2.resize((600,500))
                self.imgtk2 = ImageTk.PhotoImage(self.img_resized)
                self.label_res.configure(image = self.imgtk2)
            else:
                self.file1 = 'unidentified.png'
                self.folder1 = 'img'
                # Open dan show image
                self.dir1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join(self.folder1, self.file1))
                self.img = Image.open(self.dir1)
                self.img_resized = self.img.resize((300,400))
                self.imgtk1 = ImageTk.PhotoImage(self.img_resized)
                self.label_res.configure(image = self.imgtk1)
            self.startCamTime = time.time()

        self.label_input.after(1, self.open_cam)

    def insert_folder(self):
        self.folder = fd.askdirectory()
        extract.extract_folder(self.folder)
        eigenface.main()
        print(self.folder)

    def on_close(self, event=0):
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()
