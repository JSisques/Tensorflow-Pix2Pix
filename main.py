from tensorflow.keras.layers import *
from numpy.lib.type_check import imag
from numpy import random


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

INPATH = './inputImages/'           #Ruta de las imagenes de entrada
OUPATH = './targetImages/'          #Ruta de las imagenes de salida
CKPATH = './checkpoints/'           #Ruta de los checkpoints

SAMPLES = 500                       #Numero de muestras totales
TRAIN_SAMPLES = round(500 * 0.8)    #Numero de muestras de entrenamiento

IMG_WIDTH = 256                     #Anchura
IMG_HEIGHT = 256                    #Altura

def main():
    #Carga de datos

    imgurls = load_imgs_urls()
    randurls = suffle_list(imgurls)
    tr_urls, ts_urls = generate_partitions(randurls)
    
    train_dataset = generate_train_dataset(tr_urls)
    test_dataset = generate_test_dataset(ts_urls)
    #show_dataset_images(test_dataset)



    

def load_imgs_urls():
    #Cargamos los nombres de los ficheros
    imgs = []
    for file in os.listdir(INPATH):
        imgs.append(file)
    
    return imgs

def suffle_list(list):
    #Randomizamos la lista que contiene el nombre de las imagenes
    random_urls = np.copy(list)
    np.random.shuffle(random_urls)
    
    return random_urls
    
def generate_partitions(list):
    #Establecemos nuestros nombres de entrenamiento y test
    train_urls = list[:TRAIN_SAMPLES]
    test_urls = list[TRAIN_SAMPLES:SAMPLES]
    
    return train_urls, test_urls

def resize(inimg, tgimg, height, width):
    #Función para reescalar las imagenes

    inimg = tf.image.resize(inimg, [height, width])
    tgimg = tf.image.resize(tgimg, [height, width])

    return inimg, tgimg

def normalize(inimg, tgimg):
    #Establecemos los pixeles de las imagenes con valores entre -1 y 1
    inimg = (inimg / 127.5) - 1
    tgimg = (tgimg / 127.5) - 1

    return inimg, tgimg

def random_jitter(inimg, tgimg):
    #Cropear la imagen y hacer un flip

    inimg, tgimg = resize(inimg, tgimg, 286, 286)

    stacked_image = tf.stack([inimg, tgimg], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    inimg, tgimg = cropped_image[0], cropped_image[1]

    if tf.random.uniform(()) > 0.5:

        inimg = tf.image.flip_left_right(inimg)
        tgimg = tf.image.flip_left_right(tgimg)

    return inimg, tgimg

def load_image(filename, augment=True):
    
    inimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(INPATH + filename)), tf.float32)[..., :3] #Esta parte mantiene todas las dimensiones RGB
    tgimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(OUPATH + filename)), tf.float32)[..., :3]

    inimg, tgimg = resize(inimg, tgimg, IMG_HEIGHT, IMG_WIDTH)

    if augment:
        inimg, tgimg = random_jitter(inimg, tgimg)
    
    inimg, tgimg = normalize(inimg, tgimg)

    return inimg, tgimg

def load_train_image(filename):
    return load_image(filename, True)

def load_test_image(filename):
    return load_image(filename, False)

def show_unnormalized_image(img):
    plt.imshow((img + 1) / 2)
    plt.show()

def generate_train_dataset(train_data):
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
    train_dataset = train_dataset.map(load_train_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) #Para poder hacer procesos en paralelo eligiendo cuantos hilos por tensorflow
    train_dataset = train_dataset.batch(1) #Establecemos los lotes, en la documentacion oficial se utiliza 1

    return train_dataset

def generate_test_dataset(test_data):
    test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
    test_dataset = test_dataset.map(load_test_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) #Para poder hacer procesos en paralelo eligiendo cuantos hilos por tensorflow
    test_dataset = test_dataset.batch(1) #Establecemos los lotes, en la documentacion oficial se utiliza 1

    return test_dataset

def show_dataset_images(dataset):
    #Visualizamos las imagenes
    for inimg, tgimg in dataset.take(5):
        show_unnormalized_image(tgimg[0, ...]) # le quitamos la primera dimension que hace referencia al batch
        show_unnormalized_image(inimg[0, ...])

main()