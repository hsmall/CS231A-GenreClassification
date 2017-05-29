from scipy import misc

import numpy as np
import os
import pickle
import time

if __name__ == "__main__":
	genres = sorted([genre for genre in os.listdir("Music/") if not genre.startswith(".")])

	data = {}
	for genre in genres:
		print(genre)
		song_folders = [folder for folder in os.listdir("Music/{0}".format(genre)) if not os.path.isfile(folder)]
		for i, song_name in enumerate(song_folders):
			if i >= 60: break # Memory issues
			
			image_filenames = [file for file in os.listdir("Music/{0}/{1}".format(genre, song_name)) if file.endswith(".png")]
			image_filenames = sorted(image_filenames, key = lambda x: int(x[4:-4]))

			images = [misc.imread("Music/{0}/{1}/{2}".format(genre, song_name, filename)) / 256.0 for filename in image_filenames]
			data[song_name] = images

		with open(genre, "wb") as file:
			pickle.dump(data, file)

		data.clear()