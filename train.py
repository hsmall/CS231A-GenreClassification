from ConvolutionalNeuralNetwork import *

import argparse
import numpy as np
import os
import pickle
import random
import time

def read_data():
	genres = sorted([genre for genre in os.listdir("data/") if not genre.startswith(".")])
	
	data = {}
	for genre in genres:
		data[genre] = pickle.load(open("data/{0}".format(genre), "rb"))
	
	return data


def one_hot_vector(index, size):
	one_hot = np.zeros(size)
	one_hot[index] = 1

	return one_hot


def shuffle_pair_of_lists(one, two):
	combined = list(zip(one, two))
	random.shuffle(combined)
	one[:], two[:] = zip(*combined)
	return list(one), list(two)


def split_data(data, genres):
	x_train, y_train = [], []
	x_valid, y_valid = [], []

	for genre_index, genre in enumerate(genres):
		song_names = list(data[genre].keys())
		random.shuffle(song_names)

		one_hot = one_hot_vector(genre_index, len(genres)) 
		for song_index in range(len(song_names)):
			images = data[genre][song_names[song_index]]
			labels = [one_hot] * len(images)
			
			if song_index < 55:
				x_train.extend(images); y_train.extend(labels)
			else:
				x_valid.extend(images); y_valid.extend(labels)

	x_train, y_train = shuffle_pair_of_lists(x_train, y_train)
	x_valid, y_valid = shuffle_pair_of_lists(x_valid, y_valid)

	return x_train, y_train, x_valid, y_valid


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--num_iterations", type=int, default=10000)
	parser.add_argument("--batch_size", type=int, default=50)
	parser.add_argument("--dropout_rate", type=float, default=0.5)
	args = parser.parse_args()

	random.seed(17)
	start_time = int(round(time.time() * 1000))
	data = read_data()
	genres = sorted(list(data.keys()))
	x_train, y_train, x_valid, y_valid = split_data(data, genres)
	end_time = int(round(time.time() * 1000))
	
	print("Data Loading Time: {0} secs".format((end_time - start_time) / 1000))
	print("Number of Iterations = {0}".format(args.num_iterations))
	print("Batch Size = {0}".format(args.batch_size))
	print("Dropout Rate = {0}".format(args.dropout_rate))

	cnn = ConvolutionalNeuralNetwork(256, len(genres), args.dropout_rate)
	cnn.train(x_train, y_train, x_valid, y_valid, args.num_iterations, args.batch_size)	



