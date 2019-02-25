from imutils.video import VideoStream
import imutils
import time
from cv2 import cv2
from keras.models import load_model
from keras.preprocessing import image

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
from matplotlib import animation, style, pyplot as plt

window = tk.Tk()
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
minimum_face_confidence = 0.6

style.use('fivethirtyeight')
fig, ax = plt.subplots()

y_pos = np.arange(len(emotions))
#plt.bar(y_pos, np.zeros(len(emotions)), align='center', alpha=0.5)
plt.ion()
plt.show()
def emotions_analysis(predictions):
	plt.cla()
	predictions *= 100
	plt.bar(y_pos,predictions,align='center', alpha=0.5)
	plt.ylabel('percentage')
	plt.title('emotion')
	plt.xticks(y_pos, emotions)
	plt.yticks(np.arange(0, 110, 10))

	

# load our serialized model from disk
print("loading models...")
caffe_prototxt = 'models/face-detection/deploy.prototxt.txt'
caffe_model = 'models/face-detection/res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(caffe_prototxt, caffe_model)
model =  load_model('models/facial-expression/facial-expression-model.h5')

# initialize the video stream and allow the cammera sensor to warmup
print("starting video stream...")
vs = VideoStream(src=0).start()
#time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the face detections and facial expression
	num_of_faces = 0
	emotions_prediction = np.zeros(len(emotions))
	print(emotions_prediction)
	#print(range(0, detections.shape[2]))
	for i in range(0, detections.shape[2]):

		#get the confidence from face detection
		confidence = detections[0, 0, i, 2]
		# skip weak face detection
		if confidence < minimum_face_confidence:
			continue
		num_of_faces +=1
		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		
		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	
		face = frame[startY:endY, startX:endX]
		face = cv2.resize(face, (48,48))
		face  = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

		face_pixels = image.img_to_array(face)
		face_pixels = np.expand_dims(face_pixels, axis=0)
		face_pixels /= 255
		predictions = model.predict(face_pixels)
		emotions_prediction += predictions[0]
	
	emotions_prediction /= num_of_faces
	emotions_analysis(emotions_prediction)
		#animation.FuncAnimation(figure, emotions_analysis(predictions[0]), interval=1000)
	fig.canvas.draw()
		
	
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	#print(num_of_faces)
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

#close all windows
cv2.destroyAllWindows()
vs.stop()