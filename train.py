from dataset import Dataset
from model import Merged
from metric import MultilabelAccuracy
import tensorflow as tf
import argparse
import os

tf.keras.backend.clear_session()
tf.autograph.set_verbosity(1)
DIR_PATH = os.getcwd()

def main(args):
	
	train_images_path = os.path.join(DIR_PATH, args.train_images)
	train_texts_path = os.path.join(DIR_PATH, args.train_texts)
	train_labels_path = os.path.join(DIR_PATH, args.train_labels)
	
	print('Creating training data...', end = '')
	train = Dataset(images_path=train_images_path, texts_path=train_texts_path,\
					image_size=(args.image_width,args.image_height), labels_path=train_labels_path)
	dataset = train(training = True)
	print('done')

	if args.validation_split:
		print('Creating validation data...', end = '')
		if args.validation_split > 1 or args.validation_split < 0.1:
			raise ValueError('Validation split should lie between 0.1 and 1')
		val_size = int(len(dataset) * args.validation_split)
		train_dataset = dataset.skip(val_size)
		val_dataset = dataset.take(val_size)
		print('done')
	else:
		train_dataset = dataset
		val_dataset = None
	
	if args.data_augmentation:
		print('Augmenting training data...', end = '')
		counter = tf.data.experimental.Counter()
		train_dataset = tf.data.Dataset.zip((train_dataset, (counter, counter)))
		train_dataset = train_dataset.map(train.data_augmentation)
		print('done')
	
	train_ds = train_dataset.map(lambda image_id, image, text, label:\
					 				(((image,text),label)))
	train_ds = train_ds.shuffle(buffer_size=1000).batch(args.batch_size)

	if val_dataset is not None:
		val_ds = val_dataset.map(lambda image_id, image, text, label:\
					 				(((image,text),label)))
		val_ds = val_ds.batch(args.batch_size)

	print(f'Dataset format --> {train_ds.element_spec}')	 
	
	print('Creating model...', end = '')
	model = Merged(image_size=(args.image_width,args.image_height))
	print('done')
	
	print('Compiling model...',end = '')
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),\
              loss=tf.keras.losses.BinaryCrossentropy(from_logits= False),\
              metrics=[MultilabelAccuracy(), tf.keras.metrics.AUC(multi_label=True, name = 'roc_auc')])
	print('done')

	save_path = os.path.join(DIR_PATH, args.save_model_dir) + os.sep
	earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
	checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path,monitor='val_roc_auc',mode='max', 
                                                verbose=1, save_best_only=True, save_weights_only=True)
	
	print('Started training...')
	history = model.fit(train_ds, epochs=args.epochs, \
			validation_data= val_ds, callbacks=[earlyStopping, checkpoint])
	print(f'Model saved at {save_path}')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	# add arguments
	parser.add_argument('--image-height', type = int, default = 224)
	parser.add_argument('--image-width', type = int, default = 224)
	parser.add_argument('--batch-size', type = int, default = 32)
	parser.add_argument('--learning-rate', type = float, default = 3e-5)
	parser.add_argument('--epochs', type = int, default = 10)
	parser.add_argument('--train-images', type = str, default = 'public_train/train')
	parser.add_argument('--data-augmentation', type = bool, default = True)
	parser.add_argument('--train-texts', type = str, default = 'public_train/train_transcriptions.json')
	parser.add_argument('--train-labels', type = str, default = 'public_train/train_emotion_labels.csv')
	parser.add_argument('--validation-split', type = float, default = 0.2)
	parser.add_argument('--save-model-dir', type = str, default = 'saved_model')
	main(parser.parse_args())