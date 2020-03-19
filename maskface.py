from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers import Input, Dense, LeakyReLU, Reshape, Conv2DTranspose
from keras.models import Model, Sequential
import os
from PIL import Image
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import cv2

def model():
    encoder = Sequential()
    encoder.add(Conv2D(64,(3,3), activation='relu', padding ='same', input_shape=(200, 200, 3)))
    encoder.add(Conv2D(64,(3,3), activation='relu'))
    encoder.add(MaxPooling2D(2,2))
    encoder.add(Conv2D(128,(3,3), activation='relu', padding ='same'))
    encoder.add(Conv2D(128,(3,3), activation='relu'))
    encoder.add(MaxPooling2D(2,2))
    encoder.add(Conv2D(256,(3,3), activation='relu', padding ='same'))
    encoder.add(Conv2D(256,(3,3), activation='relu'))
    encoder.add(Conv2D(256,(3,3), activation='relu'))
    encoder.add(MaxPooling2D(2,2))
    encoder.add(Flatten())

    image_input = Input(shape=(200,200,3))
    encoding_output = encoder(image_input)

    decoding_input = Dense(32*100*100)(encoding_output)
    decoder = LeakyReLU()(decoding_input)
    decoder = Reshape((100, 100, 32))(decoder)# 100*100*32
    decoder = Conv2D(64, 3, padding='same')(decoder)
    decoder = LeakyReLU()(decoder)
    decoder = Conv2DTranspose(64,2,strides=2, padding='same')(decoder) #200*200*64
    decoder = LeakyReLU()(decoder)
    decoder = Conv2D(128, 3, padding='same')(decoder)
    decoder = LeakyReLU()(decoder)
    decoder = Conv2D(256, 3, padding='same')(decoder)
    decoder = LeakyReLU()(decoder)

    decoding_output = Conv2D(3,5,activation='tanh', padding='same')(decoder)
    generator = Model(inputs=image_input, outputs=decoding_output)
    generator.summary()

    generator_optimizer = keras.optimizers.RMSprop(lr=0.0008,
            clipvalue=1.0,decay=1e-8)
    generator.compile(optimizer=generator_optimizer,
        loss='binary_crossentropy')
    return generator

def readImg(path):
    '''
    read a image and return an array.
    '''
    files = os.listdir(path)
    data = []
    for file in files:
        im = cv2.imreag(path + file)
        im = cv2.resize(im,(170,170))
        im = img_to_array(im)
        data.append(im)
        data = np.array(data)
    return data, files

def read_data():
    training_input_path = '/Users/chaofeng/Documents/photo/6/vs_train/input'
    training_y_path =  '/Users/chaofeng/Documents/photo/6/vs_train/y'
    #training_output_path =  '/Users/chaofeng/Documents/photo/6/vs_train/y'

    test_input_path = '/Users/chaofeng/Documents/photo/6/vs_train/input'
    #test_y_path =  '/Users/chaofeng/Documents/photo/6/vs_train/y'
    #test_output_path =  '/Users/chaofeng/Documents/photo/6/vs_train/y'

    x_train, x_files = readImg(training_input_path)
    y_train = readImg(training_y_path)

    x_test = readImg(test_input_path)
    return x_train, y_train, x_test, x_files

def training():
    generator = model()
    x_train, y_train, x_test, x_files = read_data()
    save_dir = '/Users/chaofeng/Documents/photo/6/vs_train/output/'
    
    iters = 1
    batch = 1
    start = 0
    for iter in range(iters):

        stop = start + batch
        images = x_train[start: stop]
        y = y_train[start: stop]
        loss = generator.train_on_batch(images, y)
        start += batch

        if start > len(x_train) - batch:
            start = 0

    generator.save_weights('gen.h5')   
    generated_images = generator.predict(x_train)
    for i in range(generated_images.count):
        img = image.array_to_img(generated_images[i] * 255., scale=False)
        img.save(os.path.join(save_dir, x_files[i]))
    
    """
    generator.fit(x_train, y_train, epochs=1, batch_size=10,verbose=2, shuffle=False)
    generator.save_weights('gen.h5')
    generated_images = generator.predict(x_train)
    for i in range(generated_images.count):
        img = image.array_to_img(generated_images[i] * 255., scale=False)
        img.save(os.path.join(save_dir, x_files[i]))"""

training()

    



    
    






