from dataset import Dataset
from metric import MultilabelAccuracy
import tensorflow as tf
import tensorflow_text as text
import argparse
import os
import io

tf.keras.backend.clear_session()
DIR_PATH = os.getcwd()

def main(args):
	
	test_images_path = os.path.join(DIR_PATH, args.test_images)
	test_texts_path = os.path.join(DIR_PATH, args.test_texts)
	model_path = os.path.join(DIR_PATH, args.saved_model_path) + os.sep
	result_path = os.path.join(DIR_PATH, args.result_path)
	
	print('Creating testing data...', end = '')
	test = Dataset(images_path = test_images_path, texts_path = test_texts_path, \
		image_size=(args.image_width,args.image_height))
	test_datset = test()
	print('done')

	print(f'Loading saved model from {model_path}')
	model = tf.keras.models.load_model(model_path, custom_objects={'MultilabelAccuracy':MultilabelAccuracy})

	print('Predicting probabilities for test data...')
	result = io.open(os.path.join(result_path, 'results.csv'), 'w', encoding='utf-8')
	for i, (image_id, image, text) in enumerate(test_datset):
		if i % 500 == 0:
			print(f'prediction completed for {i} images.')
		
		image = tf.expand_dims(image, axis = 0)
		text = tf.expand_dims(text, axis = 0)
		prediction = model.predict([image,text])
		line = str(i) + ','+ str(image_id.numpy().decode('utf-8')) +','+ ','.join([str(x) for x in prediction[0]]) + "\n"
		result.write(line)
	result.close()

	print(f'Results saved at {result_path}')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--image-height', type = int, default = 224)
	parser.add_argument('--image-width', type = int, default = 224)
	parser.add_argument('--saved-model-path', type = str, default='saved_model')
	parser.add_argument('--test-images', type = str, default='public_train/test')
	parser.add_argument('--test-texts', type = str, default='public_train/test_transcriptions.json')
	parser.add_argument('--result-path', type = str, default='')
	main(parser.parse_args())