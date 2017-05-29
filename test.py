from ConvolutionalNeuralNetwork import *
from scipy import misc

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns

def compute_classifications(network, genres, verbose = False):
	classifications = { genre : [] for genre in genres }
	
	for genre_index, genre in enumerate(genres):
		if genre != "Pop": continue
		print(genre)
		song_folders = [folder for folder in os.listdir("Music/{0}".format(genre)) if not os.path.isfile(folder)]
		
		for song_name in song_folders[60:]:
			image_filenames = [file for file in os.listdir("Music/{0}/{1}".format(genre, song_name)) if file.endswith(".png")]
			image_filenames = sorted(image_filenames, key = lambda x: int(x[4:-4]))

			images = np.array([misc.imread("Music/{0}/{1}/{2}".format(genre, song_name, filename)) / 256.0 for filename in image_filenames])
			vector = np.sum(np.log(network.predict(images)), axis = 0)
			print("  ", vector)
			
			classifications[genre].append(vector)

	return classifications


def compute_confusion_matrix(classifications, genres):
	confusion_matrix = np.zeros((len(genres), len(genres)))

	for genre_index, genre in enumerate(genres):
		top_1 = 0.0
		top_2 = 0.0
		top_3 = 0.0

		for vector in classifications[genre]:
			prediction_indices = [index for index, value in  sorted(enumerate(vector), key=lambda x: x[1], reverse = True)]
			prediction_index = prediction_indices[0]

			confusion_matrix[genre_index][prediction_index] += 1
			if genre_index in prediction_indices[:1]: top_1 += 1
			if genre_index in prediction_indices[:2]: top_2 += 1
			if genre_index in prediction_indices[:3]: top_3 += 1

		N = len(classifications[genre])
		print("Genre: {0:18} -> Accuracies -- Top 1: {1:.3f}, Top 2: {2:.3f}, Top 3: {3:.3f}".format(genre, top_1 / N, top_2 / N, top_3 / N))

	return confusion_matrix


if __name__ == "__main__":
	np.set_printoptions(suppress=True, precision = 3)
	genres = sorted([genre for genre in os.listdir("Music/") if not genre.startswith(".")])
	
	cnn = ConvolutionalNeuralNetwork(256, len(genres))
	cnn.load("best_model/model")

	#classifications = compute_classifications(cnn, genres, verbose=True)
	#pickle.dump(classifications, open("best_model/genre_classifications_prob", "wb"))
	classifications = pickle.load(open("best_model/genre_classifications_prob", "rb"))
	
	confusion_matrix = compute_confusion_matrix(classifications, genres)
	
	'''
	confusion_matrix = np.array([[20,  0,  0,  0,  0,  0,  0,  0],
 								 [ 2, 12,  0,  0,  0,  2,  3,  1],
 								 [ 2,  0, 11,  5,  0,  0,  1,  1],
 								 [ 0,  0,  5, 10,  0,  0,  4,  1],
 								 [ 4,  3,  1,  1,  6,  2,  2,  1],
 								 [ 0,  7,  3,  2,  0,  3,  2,  3],
 								 [ 0,  0,  0,  5,  6,  1,  8,  0],
 								 [ 1,  2,  0,  3,  1,  0,  1, 12]])
	'''

	diagonal = np.diagonal(confusion_matrix)
	precision = diagonal / np.sum(confusion_matrix, axis = 0)
	print("\nPrecisions = {0}".format(precision))
	recall = diagonal / np.sum(confusion_matrix, axis = 1)
	print("Recalls = {0}".format(recall))
	print("Overall Accuracy = {0:.3f}".format(np.sum(diagonal) / np.sum(confusion_matrix)))
	
	print("\nConfusion Matrix")
	print(confusion_matrix)
	
	labels = range(1, len(genres)+1)
	#sns.heatmap(confusion_matrix, xticklabels=genres, yticklabels=genres, square=True, cbar=False, annot=True, robust=True)
	sns.heatmap(confusion_matrix, xticklabels=labels, yticklabels=labels, square=True, cbar=False, annot=True, robust=True)
	plt.xlabel("Predicted Genre", fontsize=14); plt.xticks(rotation=0)
	plt.ylabel("True Genre", fontsize=14); plt.yticks(rotation=0)
	plt.tight_layout()
	plt.show()


	






	