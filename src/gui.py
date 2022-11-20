import os
import tkinter as tk
import tkinter.messagebox
import customtkinter as ctk
import sys
from tkinter import ttk
from tkinter import *
from tkinter import filedialog as fd
from tkinter.ttk import *
from PIL import Image, ImageTk
from threading import Thread

ctk.set_appearance_mode("System")  # Modes: system (default), light, dark
ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Face Algorithm M")
        self.geometry("1000x600")
        self.protocol("WM_DELETE_WINDOW",self.on_close)

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

        self.frame_mid.columnconfigure(0, weight=4)
        self.frame_mid.columnconfigure(1, weight=1)
        self.frame_mid.columnconfigure(2, weight=4)
        self.frame_mid.rowconfigure(0, weight=1)
        self.frame_mid.rowconfigure(1, weight=1)
        self.frame_mid.rowconfigure(2, weight=1)

        self.frame_bot.rowconfigure(0, weight=1)
        self.frame_bot.rowconfigure(1,weight=1)

        self.frame_input = ctk.CTkFrame(master = self.frame_mid, width=200, height=300, corner_radius=10)
        self.frame_res =  ctk.CTkFrame(master = self.frame_mid,  width=200, height=300, corner_radius=10)
        self.frame_button_test = ctk.CTkFrame(master = self.frame_mid, width=20, height=10, corner_radius=0, fg_color='#3B3B3B')
        self.frame_title_test = ctk.CTkFrame(master = self.frame_mid, width=20, height=10, corner_radius=0, fg_color='#3B3B3B')
        self.frame_button_res = ctk.CTkFrame(master = self.frame_mid, width=20, height=10, corner_radius=0, fg_color='#3B3B3B')
        self.frame_title_res = ctk.CTkFrame(master = self.frame_mid, width=20, height=10, corner_radius=0, fg_color='#3B3B3B')

        self.frame_input.grid(row=0,column=0, sticky='nsew', padx=10, pady=(2,0))
        self.frame_button_test.grid(row=1,column=0, sticky='nsew', padx=10, pady=0)
        self.frame_title_test.grid(row=2,column=0, sticky='nsew', padx=10, pady=0)
        self.frame_res.grid(row=0,column=2, sticky='nsew', padx=10, pady=(2,0))
        self.frame_button_res.grid(row=1,column=2, sticky='nsew', padx=10, pady=0)
        self.frame_title_res.grid(row=2,column=2, sticky='nsew', padx=10, pady=0)
        
        # print image kosong
        self.insert_button = ctk.CTkButton(master=self, text = 'Insert Image',width=20,command=self.insert_img)

        self.file1 = 'empty.png'
        self.folder1 = 'img'
        # Open dan show image
        self.dir1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join(self.folder1, self.file1))
        self.img = Image.open(self.dir1)
        self.img_resized = self.img.resize((400,300))
        self.imgtk = ImageTk.PhotoImage(self.img_resized)
        # setup label
        self.label_top = ctk.CTkLabel(master = self.frame_top, text='FACE REGONITION ALGORITHM',text_font = ('Verdana',20,'bold'), anchor='center')
        self.label_input = ctk.CTkLabel(master = self.frame_mid, image = self.imgtk)
        self.label_res = ctk.CTkLabel(master = self.frame_mid,image = self.imgtk)
        self.label_tes = ctk.CTkLabel(master = self.frame_mid,text='Test Image', text_font = ("Verdana",14, 'bold'))
        self.label_tres = ctk.CTkLabel(master = self.frame_mid,text='Closest Result',text_font = ("Verdana",14,'bold'))
        self.label_bot = ctk.CTkLabel(master = self.frame_bot, text='Execute Time(s)',text_font = ("Verdana",12,'bold'))
        
        self.label_input.place(relx =0.0,rely=0.0, anchor=tkinter.N)
        self.label_top.place(relx=0.5,rely=0.5,anchor=tkinter.CENTER)
        self.label_top.grid(row=0,column=0,padx=5,pady=2)

        self.label_input.grid(row=0, column=0,padx=0,pady=2)
        self.label_res.grid(row=0, column=2,padx=0,pady=2)
        self.label_tes.grid(row=2, column=0,padx=0,pady=0)
        self.label_tres.grid(row=2, column=2,padx=0,pady=0)

        self.label_bot.grid(row=0, column=0,padx=5,pady=2)
        

        # set up button
        self.label_top.grid(row=0,column=0,padx=10,pady=2)
        self.insert_button = ctk.CTkButton(master=self.frame_mid, text = 'Insert Image',text_font = ("Cordia",12),width=100,height=40,command=self.insert_img)
        self.insert_button.grid(row=1, column=0,padx=10)

        self.insert_folder = ctk.CTkButton(master=self.frame_mid, text = 'Insert Folder',text_font = ("Cordia",12),width=100,height=40,command=self.insert_folder)
        self.insert_folder.grid(row=1, column=2,padx=10)

        # self.file1 = 'empty.png'
        # self.folder1 = 'img'
        # # Open dan show image
        # self.dir1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join(self.folder1, self.file1))
        # self.img = Image.open(self.dir1)
        # self.img_resized = self.img.resize((400,300))
        # self.imgtk = ImageTk.PhotoImage(self.img_resized)

        # self.label_mid=ctk.CTkLabel(master=self.frame_mid,image=self.imgtk)
        # self.label_mid.pack(padx=10,pady=10)
    # Layout on the main frame

    def insert_img(self):        
        # self.frame_top = ctk.CTkFrame(self, width=600,height=400, corner_radius=10, bg="#292929")
        # self.frame_top.pack(padx=20,pady=20)
        ftypes = [('Jpg Files','*.jpg'),('Png Files','*.png')]
        self.filename = fd.askopenfilename(filetypes=ftypes)
        self.dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.filename)
        self.img2 = Image.open(self.dir)
        self.img_resized = self.img2.resize((400,300))
        self.imgtk1 = ImageTk.PhotoImage(self.img_resized)
        self.label_input = ctk.CTkLabel(master = self.frame_mid, image = self.imgtk1)
        self.label_input.grid(row=0, column=0,padx=0,pady=2)
        # self.labelimg=ctk.CTkLabel(master=self,image=self.imgtk)
        # self.labelimg.pack(padx=10,pady=10)
    
    def insert_folder(self):
        self.folder = fd.askdirectory()

    def on_close(self, event=0):
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()