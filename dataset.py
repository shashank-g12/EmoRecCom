import tensorflow as tf
import pandas as pd
import json
import re
import os

class Dataset:
	"""
	Class to Build input pipeline using tensorflow Data API
	"""
	def __init__(self, images_path, texts_path, image_size, labels_path = None):
		"""
		constructor class
		inputs: 
				- images_path : str
							path to image data directory
				- texts_path : str
							path to text file containing dialogs
				- image_size: tuple
							tuple of interger in width x height format
				- labels_path : str
							path to label file if creating dataset for training data
		"""

		self.images_path = images_path
		self.texts_path = texts_path
		self.image_size = image_size
		self.labels_path = labels_path
		if labels_path is not None:
			self.df = pd.read_csv(labels_path)

	def read_json(self, filename):
		"""
		function to read json transcriptions
		inputs: 
				- filename : str
						path to transcription file
		returns: 
				- image_id : list of str
						list of image_ids in the transcriptions file 
				- texts : list of str
						list of dialogs and narration concatenated and each entry corresponds 
						to entry in image_id
		"""
		
		with open(filename, 'r') as read_file:
			transcriptions = json.load(read_file)
		
		image_id, texts = [],[]

		for transcription in transcriptions:
			text = ''
			if transcription['dialog'] is not None:
				for dialog in transcription['dialog']:
					if type(dialog) == float:
						continue
					corrected_dialog = re.sub(" ' ", "'",dialog)
					text += corrected_dialog + ', '
			
			if transcription['narration'] is not None:
				for narration in transcription['narration']:
					corrected_narration = re.sub(" ' ", "'",narration)
					text += corrected_narration + ', '
			
			text = text[:-2]
			texts.append(text)
			image_id.append(transcription['img_id'])

		return image_id, texts
	
	def get_one_label(self, image_id):
		"""
		get the label for the corresponding id
		inputs: 
				- image_id : str
						image id to access the label
		returns:
				- label : numpy array
						numpy array of labels corresponding to the image_id
		"""

		label = self.df.loc[self.df['image_id'] == image_id, ['angry', 'disgust', 'fear', 'happy', \
                                                 'sad', 'surprise', 'neutral','other']].to_numpy().squeeze()
		return label
	
	def get_labels(self, image_id, image, text):
		"""
		function to add label to dataset
		inputs:
				- element : tf.data element
						dataset element in tuple form (image_id, image_path, text)
		returns: 
				- element : tf.data element
						dataset elemet with label added
		"""

		label = tf.py_function(self.get_one_label, inp = [image_id], Tout = tf.int8)
		label.set_shape([8,])

		return image_id, image, text, label

	def load_image(self, image_id, image_path, text):
		"""
		map function to load image
		inputs:
				- element : tf.data element
						dataset element in tuple form (image_id, image_path, text)
		returns: 
				- element : tf.data element
						dataset elemet with image in image_path
		"""
		
		image = tf.io.read_file(image_path)
		image = tf.image.decode_jpeg(image, channels=3)
		
		return image_id, image, text

	def preprocess(self, image_id, image, text):
		"""
		function to preprocess image
		inputs:
				- element : tf.data element
						dataset element in tuple form (image_id, image_path, text)
		returns: 
				- element : tf.data element
						dataset elemet with image processed
		"""

		image = tf.cast(image, tf.float32)
		image = tf.keras.applications.resnet.preprocess_input(image)
		image = tf.image.resize(image, [self.image_size[1], self.image_size[0]])
		
		return image_id, image, text
	
	def random_rotation(self, image, p = 0.5):
		"""
		perfroms rotation on image
		inputs:
				- image ; tensor
						input image
				- p : float
						probability between 0 and 1
		returns: 
				- image : tensor 
						rotated image
		"""

		if p < 0 or p > 1:
			raise ValueError("probability should be between 0 and 1")
  		
		if(tf.random.uniform([]) < p):
   	 		image = tf.keras.preprocessing.image.random_rotation(image.numpy(), 20,0,1,2)
		
		return image

	def data_augmentation(self, element, seed):
		"""
		performs data aumentation on image
		inputs:
				- element : tf.data element
						dataset element in tuple form (image_id, image_path, text)
				- seed : tuple of int
						seed to perfrom random augmentation
		returns: 
				- element : tf.data element
						dataset elemet with image augmented
		"""

		image_id, image, text , label = element

  		# random horizontal_flip
		new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
		image = tf.image.stateless_random_flip_left_right(image, seed = new_seed)
		
		# random rotation (20 degrees)
		im_shape = image.shape
		image = tf.py_function(self.random_rotation, inp = [image], Tout = tf.float32)
		image.set_shape(im_shape)

		return image_id, image, text , label

	def __call__(self, training = False):
		"""
		function to create dataset
		inputs:
				- training : bool
						indicating whether dataset being created is for training or testing
		returns:
				- dataset : tf.data
						the dataset preprocessed according to the given arguments
		"""

		image_id, texts = self.read_json(self.texts_path)
		all_images_path  = [os.path.join(self.images_path,id+'.jpg') for id in image_id]

		dataset = tf.data.Dataset.from_tensor_slices((image_id, all_images_path, texts))
		dataset = dataset.map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE)	
		dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
		dataset = dataset.map(self.preprocess)
		
		if training:
			dataset = dataset.map(self.get_labels)

		return dataset