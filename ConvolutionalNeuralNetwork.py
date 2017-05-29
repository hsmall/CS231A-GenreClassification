import numpy as np
import os
import random
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress Logging/Warnings


class ConvolutionalNeuralNetwork:
	def __init__(self, image_size, num_classes, dropout_rate=0.5):
		self.image_size = image_size
		self.num_classes = num_classes

		self.is_training = tf.placeholder(tf.bool)
		self.x = tf.placeholder(tf.float32, [None, image_size, image_size, 1])
		self.y = tf.placeholder(tf.float32, [None, num_classes])

		self.conv1 = tf.layers.conv2d(inputs = self.x,
								      filters = 32,
								      kernel_size = 3,
								      padding = "same",
								      kernel_initializer = tf.contrib.layers.xavier_initializer(),
								      bias_initializer = tf.contrib.layers.xavier_initializer(),
								      activation = tf.nn.elu,
								      name = "conv1")

		self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1,
											 pool_size = 4,
											 strides = 2,
											 name = "pool1")
		'''
		self.pool1_dropout = tf.layers.dropout(inputs = self.pool1,
										 rate = dropout_rate,
										 training = self.is_training,
										 name = "pool1_dropout") 
		'''

		self.conv2 = tf.layers.conv2d(inputs = self.pool1,
								      filters = 64,
								      kernel_size = 3,
								      padding = "same",
								      kernel_initializer = tf.contrib.layers.xavier_initializer(),
								      bias_initializer = tf.contrib.layers.xavier_initializer(),
								      activation = tf.nn.elu,
								      name = "conv2")

		self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2,
											 pool_size = 4,
											 strides = 2,
											 name = "pool2")
		'''
		self.pool2_dropout = tf.layers.dropout(inputs = self.pool2,
										 rate = dropout_rate,
										 training = self.is_training,
										 name = "pool2_dropout")
		'''

		self.conv3 = tf.layers.conv2d(inputs = self.pool2,
								      filters = 128,
								      kernel_size = 3,
								      padding = "same",
								      kernel_initializer = tf.contrib.layers.xavier_initializer(),
								      bias_initializer = tf.contrib.layers.xavier_initializer(),
								      activation = tf.nn.elu,
								      name = "conv3")

		self.pool3 = tf.layers.max_pooling2d(inputs=self.conv3,
											 pool_size = 4,
											 strides = 2,
											 name = "pool3")
		'''
		self.pool3_dropout = tf.layers.dropout(inputs = self.pool3,
										 rate = dropout_rate,
										 training = self.is_training,
										 name = "pool3_dropout")
		'''

		self.conv4 = tf.layers.conv2d(inputs = self.pool3,
								      filters = 256,
								      kernel_size = 3,
								      padding = "same",
								      kernel_initializer = tf.contrib.layers.xavier_initializer(),
								      bias_initializer = tf.contrib.layers.xavier_initializer(),
								      activation = tf.nn.elu,
								      name = "conv4")

		self.pool4 = tf.layers.max_pooling2d(inputs=self.conv4,
											 pool_size = 2,
											 strides = 2,
											 name = "pool4")
		'''
		self.pool4_dropout = tf.layers.dropout(inputs = self.pool4,
										 rate = dropout_rate,
										 training = self.is_training,
										 name = "pool4_dropout")
		'''

		self.dense1 = tf.layers.dense(inputs = tf.reshape(self.pool4, [-1, self.tuple_product(self.pool4.shape)]),
									 units = 1024,
									 kernel_initializer = tf.contrib.layers.xavier_initializer(),
									 bias_initializer = tf.contrib.layers.xavier_initializer(),
									 activation = tf.nn.elu,
									 name = "dense1")
		
		self.dropout1 = tf.layers.dropout(inputs = self.dense1,
										 rate = dropout_rate,
										 training = self.is_training,
										 name = "dropout1")

		self.dense2 = tf.layers.dense(inputs = tf.reshape(self.dropout1, [-1, self.tuple_product(self.pool4.shape)]),
									 units = 1024,
									 kernel_initializer = tf.contrib.layers.xavier_initializer(),
									 bias_initializer = tf.contrib.layers.xavier_initializer(),
									 activation = tf.nn.elu,
									 name = "dense2")
		
		self.dropout2 = tf.layers.dropout(inputs = self.dense1,
										 rate = dropout_rate,
										 training = self.is_training,
										 name = "dropout2")

		self.output = tf.layers.dense(inputs = self.dropout2,
  									  units = self.num_classes,
  									  name = "output")

		self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = self.output))
		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.output, axis = 1), tf.argmax(self.y, axis = 1)), tf.float32) )

		self.train_step = tf.train.RMSPropOptimizer(1e-4).minimize(self.error)

		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())
		

	def tuple_product(self, tup):
		product = 1
		for elem in tup:
			try:
				product *= int(elem)
			except:
				pass
		return product

	def save(self, filename):
		self.saver = tf.train.Saver()
		self.saver.save(self.session, filename)

	def load(self, filename):
		self.saver = tf.train.Saver()
		self.saver.restore(self.session, filename)

	def predict(self, data):
		data_reshaped = np.reshape(data, [-1, self.image_size, self.image_size, 1])
		output = self.session.run(tf.nn.softmax(self.output), feed_dict = {self.x: data_reshaped, self.is_training: False})
		return output
		#return np.argmax(output, axis = 1)

	def train(self, x_train, y_train, x_valid, y_valid, num_iterations = 3000, batch_size = 20):
		print ("Training Set Size: {0}, Validation Set Size: {1}".format(len(y_train), len(y_valid)))
		x_train_reshaped = np.reshape(x_train[:500], [-1, self.image_size, self.image_size, 1])
		x_valid_reshaped = np.reshape(x_valid[:500], [-1, self.image_size, self.image_size, 1])

		lowest_error = float("inf")
		lowest_error_iteration = 0
		for iteration in range(num_iterations + 1):
			x_batch, y_batch = zip(*random.sample(list(zip(x_train, y_train)), batch_size))
			x_batch = np.reshape(x_batch, [-1, self.image_size, self.image_size, 1])

			if iteration % 50 == 0:
				train_error, train_accuracy = self.session.run((self.error, self.accuracy),
															   feed_dict = {self.x: x_train_reshaped,
															            	self.y: y_train[:500],
															            	self.is_training: False})
				
				valid_error, valid_accuracy = self.session.run((self.error, self.accuracy),
															   feed_dict = {self.x: x_valid_reshaped,
															            	self.y: y_valid[:500],
															            	self.is_training: False})
				marker = ""
				if valid_error <= lowest_error:
					lowest_error = valid_error
					lowest_error_iteration = iteration
					self.save("model")
					marker = "*"

				print("Iteration: {0:>4} -> Train: {1:.3f} ({2:.3f}), Validation: {3:.3f} ({4:.3f}) {5}".format(iteration,
																											train_error,
																											train_accuracy,
																											valid_error,
																											valid_accuracy,
																											marker))
				
			
			self.session.run(self.train_step, feed_dict = {self.x: x_batch,
														   self.y: y_batch,
														   self.is_training: True})



