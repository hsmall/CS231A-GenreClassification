from scipy.io import wavfile
from scipy.signal import spectrogram

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import time

def convert_to_image(matrix):
	image_min = np.amin(matrix)
	image_max = np.amax(matrix)
	image = (((matrix - image_min) / (image_max - image_min)) * 255.9).astype(np.uint8)
	return Image.fromarray(image)

genres = sorted([genre for genre in os.listdir("Music/") if not genre.startswith(".")])

start = int(round(time.time() * 1000))
for genre in genres:
	count = 0
	song_folders = [folder for folder in os.listdir("Music/{0}".format(genre)) if not os.path.isfile(folder)]
	song_folders = sorted(song_folders)
	print(genre, len(song_folders))
	
	for i, song_name in enumerate(song_folders[:]):
		print("  {0}) {1}".format(i+1, song_name))

		# Read in .wav file
		rate, audio = wavfile.read("Music/{0}/{1}/{1}.wav".format(genre, song_name))
		if len(audio.shape) > 1: audio = np.mean(audio, axis = 1)

		# Convert .wav to a spectrogram
		length = 512
		freqs, times, Sxx = spectrogram(audio, fs = rate,
										window = 'hanning',
										nperseg= length,
										detrend = False,
										scaling = 'spectrum')
		Sxx = 10 * np.log10(Sxx + np.full(Sxx.shape, 1e-100))

		# Split spectrogram into 256-by-256 chunks
		image_size = length//2
		num = 1
		for start_index in range(0, Sxx.shape[1], image_size):
			image = Sxx[:image_size, start_index:start_index + image_size]
			if image.shape != (image_size, image_size): continue

			image = convert_to_image(image.copy())
			image.save("Music/{0}/{1}/img_{2}.png".format(genre, song_name, num))
			num += 1

end = int(round(time.time() * 1000))
print((end-start)/1000)

# from pydub import AudioSegment
# sound = AudioSegment.from_mp3(mp3_filename)
# sound.export(wav_filename, format="wav")
