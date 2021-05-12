import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

def BERT():
	"""
	creates a functional model for processing text data
	returns:
			- model : tf.Model
					model for processing text
	"""

	text_input = tf.keras.Input(shape = (), dtype = tf.string, name = 'text')
	preprocessing_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3', name='preprocessing')
	encoder_inputs = preprocessing_layer(text_input)
	encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3', trainable=True, name='BERT_encoder')
	outputs = encoder(encoder_inputs)
	net = outputs['pooled_output']

	return tf.keras.Model(inputs = text_input, outputs = net)


def ResNet50(image_size):
	"""
	creates a functional model for processing image data
	returns:
			- model : tf.Model
					model for processing image
	"""

	ResNet_model = tf.keras.applications.ResNet50(include_top = False, weights = 'imagenet', input_shape = (224, 224, 3))
	image_input = tf.keras.Input(shape = (image_size[1], image_size[0], 3), name = 'image')
	image_features = ResNet_model(image_input)
	global_pooling = tf.keras.layers.GlobalAveragePooling2D(name = 'pooling')(image_features)
	dropout = tf.keras.layers.Dropout(0.5)(global_pooling)

	return tf.keras.Model(inputs = image_input, outputs = dropout)

def Merged(image_size):
	"""
	creates a functional model by merging the model for text and image
	returns:
			- model : tf.Model
					model for processing text and image
	"""
	
	ResNet = ResNet50(image_size)
	Bert = BERT()
	
	text_input = tf.keras.Input(shape = (), dtype = tf.string, name = 'text')
	image_input = tf.keras.Input(shape = (image_size[1], image_size[0], 3), name = 'image')
	image_features = ResNet(image_input)
	text_features = Bert(text_input)
	output = tf.keras.layers.concatenate([image_features, text_features], axis = -1)
	output = tf.keras.layers.Dense(128, activation='relu')(output)
	output = tf.keras.layers.Dense(8, activation='sigmoid')(output)

	return tf.keras.Model(inputs = [image_input, text_input], outputs = output)