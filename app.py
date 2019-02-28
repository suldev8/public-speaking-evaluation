import tkinter as tk
from cv2 import cv2
from PIL import Image, ImageTk
import time
from imutils.video import VideoStream
from emotion_recognition import emotion_predictions
import numpy as np
from matplotlib import animation, style, pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

style.use('fivethirtyeight')
fig, ax = plt.subplots()
emotions_predictions = np.zeros(len(emotions))
y_pos = np.arange(len(emotions))
# plt.bar(y_pos, np.zeros(len(emotions)), align='center', alpha=0.5)

def emotions_analysis(i):
    global emotions_predictions
    emotions_predictions *= 100
    print(emotions_predictions)
    ax.clear()
    plt.bar(y_pos, emotions_predictions, align='center', alpha=0.5)
    plt.ylabel('percentage')
    plt.title('emotion')
    plt.xticks(y_pos, emotions)
    plt.yticks(np.arange(0, 110, 10))

window = tk.Tk()
window.title("Public Speaking Evaluation")
video = cv2.VideoCapture(0)
if not video.isOpened():
    raise ValueError("unable to open video")
videoStreamFrame = tk.Frame(window)
videoStreamFrame.grid()
videoLabel = tk.Label(videoStreamFrame)
videoLabel.grid()

def video_stream():
    global emotions_predictions
    _, frame = video.read()
    frame, emotions_predictions = emotion_predictions(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = ImageTk.PhotoImage(image = Image.fromarray(frame), master=videoStreamFrame)
    videoLabel.imgtk = image
    videoLabel.config(image=image)
    videoLabel.after(40,video_stream)


video_stream()
plotCanvas = FigureCanvasTkAgg(fig, window)
plotCanvas.get_tk_widget().grid()
ani = animation.FuncAnimation(fig, emotions_analysis, interval=500, blit=False)
window.mainloop()

""" class MainView():
    def __init__(self, window, title, VideoSource):
        self.window = self.window
        self.window.title(title)
        self.video = VideoCapture(0)

        self.window.mainloop()

class VideoCapture():
    def __init__(self, video_source):
        self.video = cv2.VideoCapture(video_source)
        if not video.isOpened():
            raise ValueError("unable to open video")
    def 
if __name__ == "__main__":
    MainView(tk.Tk(), "Public Speaking Evaluation", 0)
    window.mainloop() """