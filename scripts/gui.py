import os
from tkinter import *
from PIL import ImageTk,Image as im
import numpy as np
import matplotlib.pyplot as plt
from main import TrainModel
from data_preprocessing import DataPreprocess

tmodel = TrainModel()
model = tmodel.loadmodel()

data = DataPreprocess()
x_test, y_test, _, _ = data.load_data('test.pickle')
x_test, y_test = data.shuffle_data(x_test, y_test)
x_test_norm = data.normalize_data(x_test)
label_list = data.label_list


def save_test_image():
    # load test data

    for i in range(100):    

        plt.imsave('test_images/test'+ str(i)+'.png', x_test_norm[i,:,:,:])



root = Tk()
root.title('Traffic Sign Classification')

image_list = []
# images = sorted(os.listdir('test_images/'))
for i in range(100):

	
	file = 'test_images/test' + str(i) + '.png'
	# print('img:', file)
	img1 = im.open(file)
	img1 = ImageTk.PhotoImage(img1)
	img1 = img1._PhotoImage__photo.zoom(7)
	image_list.append(img1)

# image_list.sort()
# print(image_list)


status = Label(root, text="Image 1 of " + str(len(image_list)), bd=1, relief=SUNKEN, anchor=E)
pred = Label(root, text="prediction:  ", bd=1, relief=SUNKEN, anchor=E)
my_label = Label(image=image_list[0])
my_label.grid(row=0, column=0, columnspan=5)

image_number = 1

def forward(imn):
	global image_number
	global my_label
	global button_forward
	global button_back

	image_number = imn
	my_label.grid_forget()
	my_label = Label(image=image_list[image_number-1])
	button_forward = Button(root, text=">>", command=lambda: forward(image_number+1))
	button_back = Button(root, text="<<", command=lambda: back(image_number-1))
	
	if image_number == 100:
		button_forward = Button(root, text=">>", state=DISABLED)

	my_label.grid(row=0, column=0, columnspan=3)
	# button_predict.grid(row=1)
	button_back.grid(row=3, column=0)
	button_forward.grid(row=3, column=2, pady=10)

	pred = Label(root, text="prediction:  ", bd=1, relief=SUNKEN, anchor=E)
	pred.grid(row=2, column=0, columnspan=3, sticky=W+E)

	status = Label(root, text="Image " + str(image_number) + " of " + str(len(image_list)), bd=1, relief=SUNKEN, anchor=E)
	status.grid(row=4, column=0, columnspan=3, sticky=W+E)
	
		

def back(imn):
	global image_number
	global my_label
	global button_forward
	global button_back

	image_number = imn
	my_label.grid_forget()
	my_label = Label(image=image_list[image_number-1])
	button_forward = Button(root, text=">>", command=lambda: forward(image_number+1))
	button_back = Button(root, text="<<", command=lambda: back(image_number-1))

	if image_number == 1:
		button_back = Button(root, text="<<", state=DISABLED)

	my_label.grid(row=0, column=0, columnspan=3)
	# button_predict.grid(row=1)
	button_back.grid(row=3, column=0)
	button_forward.grid(row=3, column=2, pady=10)

	pred = Label(root, text="prediction:  ", bd=1, relief=SUNKEN, anchor=E)
	pred.grid(row=2, column=0, columnspan=3, sticky=W+E)

	# Update Status Bar
	status = Label(root, text="Image " + str(image_number) + " of " + str(len(image_list)), bd=1, relief=SUNKEN, anchor=E)
	status.grid(row=4, column=0, columnspan=3, sticky=W+E)


def predict_sign():
	
	res = model.predict(x_test_norm[image_number-1:image_number, :, :, :])
	c = np.argmax(res)
	# data.show_image(x_test_norm[image_number-1, :, :, :], y_test[image_number-1])
	prediction = label_list[c]
	# print('image_number: ',image_number)

	pred = Label(root, text="prediction:  " + str(prediction), bd=1, relief=SUNKEN, anchor=E)
	pred.grid(row=2, column=0, columnspan=3, sticky=W+E)


button_predict = Button(root, text='Predict', command = predict_sign)
button_back = Button(root, text="<<", command=back, state=DISABLED)
button_exit = Button(root, text="Exit", command=root.quit)
button_forward = Button(root, text=">>", command=lambda: forward(2))

button_predict.grid(row=1, column=1)
pred.grid(row=2, column=0, columnspan=3, sticky=W+E)
button_back.grid(row=3, column=0)
button_exit.grid(row=3, column=1)
button_forward.grid(row=3, column=2, pady=10)
status.grid(row=4, column=0, columnspan=3, sticky=W+E)


root.mainloop()