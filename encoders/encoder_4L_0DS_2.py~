from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
def get_lr_encoder():
	input_img = Input(shape=(128, 128, 1)) 
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
	encoded = MaxPooling2D((2, 2), padding='same')(x)

	x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	test = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(test)
	x = UpSampling2D((2, 2))(x)
	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
	lr_model = Model(input_img, decoded)
	lr_model.compile(optimizer='adadelta', loss='binary_crossentropy')
	return lr_model 
