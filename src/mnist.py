#!/usr/bin/env python

"""
Load and manipulate the MNIST database of handwritten digits.

The default paths assume that the database is stored in /data/mnist as raw binary files.

/data/
	/mnist/
		/training_images
		/training_labels
		/test_images
		/test_labels
/src/
	/mnist.py

Information on the MNIST database is found at http://yann.lecun.com/exdb/mnist/
"""

import struct
import Tkinter

def load_training_labels(db_location="..\\data\\mnist\\training_labels"):
	"""
	Return a list of labels in database order.

	Labels are integers.
	"""
	with open(db_location, "rb") as f:
		# Check magic number.
		assert struct.unpack(">I", f.read(4))[0] == 2049

		# Get number of labels.
		label_count = struct.unpack(">I", f.read(4))[0]

		# Read that many labels.
		labels = []
		for i in xrange(label_count):
			label = struct.unpack(">B", f.read(1))[0]
			labels.append(label)

		return labels

def load_training_images(db_location="..\\data\\mnist\\training_images"):
	"""
	Return a list of images in database order.

	Images are a 784-dimensional tuple of values in [0, 255].
	"""
	with open(db_location, "rb") as f:
		# Check magic number.
		assert struct.unpack(">I", f.read(4))[0] == 2051

		# Get number of images.
		image_count = struct.unpack(">I", f.read(4))[0]

		# Get number of rows.
		row_count = struct.unpack(">I", f.read(4))[0]

		# Get number of columns.
		column_count = struct.unpack(">I", f.read(4))[0]

		# Read pixels.

		# We will read batches of images to minimise file operations.
		total_images_to_read = image_count
		batch_size = 10000 # Totally arbitrary.
		images = []

		while total_images_to_read > 0:
			if total_images_to_read > batch_size:
				images_to_read = batch_size
			else:
				images_to_read = total_images_to_read

			data = f.read(images_to_read*row_count*column_count)
			for im in xrange(images_to_read):
				image = []
				for px in xrange(row_count*column_count):
					pixel = struct.unpack(">B", data[im*row_count*column_count+px])[0]
					image.append(pixel)
				images.append(tuple(image))

			total_images_to_read -= images_to_read

		assert len(images) == image_count

		return images

def view_image(image, width, height):
	"""
	Open a Tkinter window to view an image from the MNIST dataset.
	"""
	root = Tkinter.Tk()
	root.minsize(width, height)
	root.geometry("{}x{}".format(width*2, height*2))
	root.bind("<Button>", lambda e: e.widget.quit())
	im = Tkinter.PhotoImage(width=width, height=height)
	im.put(" ".join(("{" + " ".join("#{:02x}{:02x}{:02x}".format(image[i+width*j],image[i+width*j],image[i+width*j]) for i in xrange(width)) + "}") for j in xrange(height)))
	w = Tkinter.Label(root, image=im)
	w.pack(fill="both", expand=True)
	root.mainloop()