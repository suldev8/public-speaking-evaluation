import tkinter as tk
from tkinter import ttk, filedialog

from PIL import ImageTk, Image
from cv2 import cv2
from emotion_recognition import get_emotion_predictions

import matplotlib
from matplotlib import style, pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np

matplotlib.use("TkAgg")

LARGE_FONT = ("", 12)

video_path = 0

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
style.use('fivethirtyeight')

y_pos = np.arange(len(emotions))

class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.iconbitmap(self, default="assets/app_icon.ico")
        tk.Tk.title(self, "Public Speaking Evaluation")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for f in (StartPage, EmotionRecognitionPage):
            frame = f(container, self)
            self.frames[f] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        logo = ImageTk.PhotoImage(Image.open("assets/app_icon.png"), master=self)
        logo_label = ttk.Label(self, image=logo)
        logo_label.image = logo
        logo_label.pack()

        start_label = ttk.Label(self, text="Load a video or click stream to analyze emotions",
                                font=LARGE_FONT)
        start_label.pack(pady=10, padx=10)
        btn_frame = ttk.Frame(self)
        btn_frame.pack()

        load_btn = ttk.Button(btn_frame, text="Load Video",
                              command=lambda: self.load_video(controller))
        load_btn.pack(pady=10, padx=10, side="left")

        stream_btn = ttk.Button(btn_frame, text="Streaming",
                                command=lambda: self.streaming(controller))
        stream_btn.pack(pady=10, padx=10, side="left")

    def load_video(self, controller):
        global video_path
        filetypes = [
            ("Video Files", ("*.mp4", "*.avi")),
            ("MP4", '*.mp4'),
            ("AVI", '*.avi'),
            ('All', '*')
        ]
        video_path = filedialog.askopenfilename(
            title="Select video", filetypes=filetypes)

        if video_path:
            controller.show_frame(EmotionRecognitionPage)
    def streaming(self, controller):
        global video_path
        video_path = 0
        controller.show_frame(EmotionRecognitionPage)


class EmotionRecognitionPage(tk.Frame):
    def __init__(self, parent, controller):
        
        tk.Frame.__init__(self, parent)
        btn_frame = ttk.Frame(self)
        btn_frame.pack(side="top")
        
        back_btn = ttk.Button(
            btn_frame, text="Back", command=lambda: self.back_to_start_page(controller))
        back_btn.pack(pady=10, padx=10, side="left")
        
        self.start_stop_btn = ttk.Button(
            btn_frame, text="Start", command=self.start_stop)
        self.start_stop_btn.pack(pady=10, padx=10, side="right")
        
        self.video = None
        self.video_frame = tk.Frame(self)
        self.video_frame.pack()
        self.video_label = tk.Label(self.video_frame)
        
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.10)
        self.bars = self.ax.bar(y_pos, np.zeros(len(emotions)), align='center', alpha=0.5)
        self.ax.set_ylabel('percentage')
        self.ax.set_title('emotion')
        self.ax.set_xticks(y_pos)
        self.ax.set_xticklabels(emotions)
        self.ax.set_yticks(np.arange(0, 110, 10))
        
        self.canvas_bar = FigureCanvasTkAgg(self.fig, self.video_frame)
        self.canvas_bar.draw()
        
        self.average_emotion = np.zeros(len(emotions))
        self.numb_of_frames = 0

    def start_stop(self):
        if self.start_stop_btn.cget("text") == "Start":
            self.start_stop_btn.config(text="Stop")
            self.video_label.pack(pady=10, padx=10, side="left")
            self.canvas_bar.get_tk_widget().pack(pady=10, padx=10, side="right", fill="both", expand="1")
            self.start_video()
        
        elif self.start_stop_btn.cget("text") == "Stop":
            self.video.__del__()
            self.start_stop_btn.config(text="Start")
            self.video_label.forget()
            self.canvas_bar.get_tk_widget().forget()
            self.show_average()
    
    def show_average(self):
        self.canvas_bar.get_tk_widget().pack(pady=10, padx=10, side="top")
        self.average_emotion /= self.numb_of_frames
        self.draw_bar_chart(self.average_emotion)

    def start_video(self):
        global video_path
        self.video = VideoCapture(video_path)
        self.stream_video()

    def stream_video(self):
        ret, frame = self.video.get_frame()
        if ret:
            frame, emotion_predictions = get_emotion_predictions(frame)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = ImageTk.PhotoImage(image = Image.fromarray(frame), master=self.video_frame)
            self.video_label.imgtk = image
            self.video_label.config(image=image)
            if(emotion_predictions.all(0)):
                self.average_emotion += emotion_predictions
                self.numb_of_frames += 1
            self.draw_bar_chart(emotion_predictions)
            
            print(self.average_emotion)
            print(self.numb_of_frames)
            self.video_label.after(100, self.stream_video)
        else:
            self.video_label.forget()
    
    def draw_bar_chart(self, emotion_predictions):

        emotion_predictions *= 100
        for rect, h in zip(self.bars, emotion_predictions):
            rect.set_height(h)
        self.canvas_bar.draw()

    def back_to_start_page(self, controller):
        if self.video:
            self.video.__del__()
        self.start_stop_btn.config(text="Start")
        self.video_label.forget()
        controller.show_frame(StartPage)

class VideoCapture:
    def __init__(self, video_path=0):
        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            raise ValueError("Couldn't open video file", video_path)
    def get_frame(self):
        if self.video.isOpened():
            ret, frame = self.video.read()
            if ret:
                return (ret, frame)
            else:
                return (ret, None)
        else:
            return(False, None)
    
    def __del__(self):
        if self.video.isOpened():
            self.video.release()

if __name__ == "__main__":
    App().mainloop()
