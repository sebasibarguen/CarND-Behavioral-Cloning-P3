
import os
import argparse
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D

def preprocess_images(images):
    pass


def get_commaai_model():
	"""
	Creates the comma.ai model, and returns a reference to the model
	The comma.ai model's original source code is available at:
	https://github.com/commaai/research/blob/master/train_steering_model.py
	"""
	ch, row, col = CH, H, W  # camera format

	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.,
		input_shape=(row, col, ch),
		output_shape=(row, col, ch)))
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='same', W_regularizer=l2(L2_REG_SCALE)))
	model.add(ELU())
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='same', W_regularizer=l2(L2_REG_SCALE)))
	model.add(ELU())
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='same', W_regularizer=l2(L2_REG_SCALE)))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())
	model.add(Dense(512, W_regularizer=l2(0.)))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(1, W_regularizer=l2(0.)))

	model.compile(optimizer=Adam(lr=LR), loss='mean_squared_error')

	return model


def train():
    pass
