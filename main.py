from tensorflow.keras.layers import *
from numpy.lib.type_check import imag
from numpy import random

import matplotlib.pyplot as plt
import tensorflow.keras as kr
import tensorflow as tf
import numpy as np
import os

from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.ops.gen_array_ops import shape

INPATH = './inputImages/'           #Ruta de las imagenes de entrada
OUPATH = './targetImages/'          #Ruta de las imagenes de salida
CKPATH = './checkpoints/'           #Ruta de los checkpoints

SAMPLES = 500                       #Numero de muestras totales
TRAIN_SAMPLES = round(500 * 0.8)    #Numero de muestras de entrenamiento

IMG_WIDTH = 256                     #Anchura
IMG_HEIGHT = 256                    #Altura

LAMBDA = 100

def main():
    #Carga de datos

    imgurls = load_imgs_urls()
    randurls = suffle_list(imgurls)
    tr_urls, ts_urls = generate_partitions(randurls)
    
    train_dataset = generate_train_dataset(tr_urls)
    test_dataset = generate_test_dataset(ts_urls)
    #show_dataset_images(test_dataset)

    #Modelo
    model_generator = generator()
    model_discriminator = discriminator()

    #Funciones de coste

   

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

def downsample(filters, apply_batchnorm = True):
    result = kr.Sequential()

    initializer = tf.random_normal_initializer(0, 0.02)     #Ruido Gaussiano y derivada

    #Capa convolucional
    result.add(Conv2D(filters, 
    kernel_initializer=initializer, 
    kernel_size=4, 
    strides=2, #Para dividir el tamaño a la mitad
    padding='same',
    use_bias=not apply_batchnorm
    ))

    if apply_batchnorm:
        #Capa BatchNorm
        result.add(BatchNormalization())

    #Capa de activacion
    result.add(LeakyReLU())


    return result

def upsample(filters, apply_dropout = False):
    result = kr.Sequential()

    initializer = tf.random_normal_initializer(0, 0.02)     #Ruido Gaussiano y derivada

    #Capa convolucional inversa
    result.add(Conv2DTranspose(filters, 
    kernel_initializer=initializer, 
    kernel_size=4, 
    strides=2, 
    padding='same',
    use_bias=False
    ))

     #Capa BatchNorm
    result.add(BatchNormalization()) 

    if apply_dropout:
       result.add(Dropout(0.5))

    #Capa de activacion
    result.add(ReLU())


    return result

def generator():

    inputs = tf.keras.layers.Input(shape=[None, None, 3]) #Para que nuestra imagen pueda tener diferenetes tamaños

    down_stack = [
        downsample(64, apply_batchnorm=False),      #(bs,   128,    128,    64)
        downsample(128),                            #(bs,   64,     64,     128)
        downsample(256),                            #(bs,   32,     32,     256)
        downsample(512),                            #(bs,   16,     16,     512)
        downsample(512),                            #(bs,   8,      8,      512)
        downsample(512),                            #(bs,   4,      4,      512)
        downsample(512),                            #(bs,   2,      2,      512)
        downsample(512),                            #(bs,   1,      1,      512)
    ]

    up_stack = [
        upsample(512, apply_dropout=True),          #(bs,   2,      2,      1024)
        upsample(512, apply_dropout=True),          #(bs,   4,      4,      1024)
        upsample(512, apply_dropout=True),          #(bs,   8,      8,      1024)
        upsample(512),                              #(bs,   16,     16,     1024)
        upsample(256),                              #(bs,   32,     32,     512)
        upsample(128),                              #(bs,   64,     64,     256)
        upsample(64),                               #(bs,   128,    128,    128)
    ]

    initializer = tf.random_normal_initializer(0, 0.02)     #Ruido Gaussiano y derivada

    last = Conv2DTranspose(
        filters=3, #Ya que queremos una imagen RGB
        kernel_size=4,
        strides=2, #Para aumentar el doble el tamaño
        padding='same',
        kernel_initializer=initializer,
        activation='tanh' #Ya que queremos una salida de -1 a 1
    )

    #Conectamos todas las capas del codificador
    x = inputs
    s = []      #Skip conncetions

    concat = Concatenate()

    for down in down_stack:
        x = down(x)
        s.append(x)
    
    #Conectamos todas las capas del decodificador
    s = reversed(s[:-1]) #El ultimo elemento no nos hace falta
    for up, sk in zip(up_stack, s):
        x = up(x)
        x = concat([x, sk])
    
    last = last(x)
    model = kr.Model(inputs=inputs, outputs=last)
    model.summary()
    return model

def discriminator():

    ini = Input(shape=[None, None, 3], name='input_img')
    gen = Input(shape=[None, None, 3], name='gener_img')

    con = concatenate([ini, gen])

    initializer = tf.random_normal_initializer(0, 0.02)

    down1 = downsample(64, apply_batchnorm=False)(con)
    down2 = downsample(128)(down1)
    down3 = downsample(256)(down2)
    down4 = downsample(512)(down3)

    last = Conv2D(
        filters=1,
        kernel_size=4,
        strides=1,
        kernel_initializer=initializer,
        padding='same'
    )(down4)

    model = kr.Model(inputs = [ini, gen], outputs=last)

    model.summary()

    return model

def set_losses():
    #from_logits=True => para que las imagenes se normalicen pasandolas por una funcion sigmoide para que se queden entre 0 y 1 
    loss_object = kr.losses.BinaryCrossentropy(from_logits=True)
    total_disc_loss = discriminator_loss(loss_object)

def discriminator_loss(loss_object, disc_real_output, disc_generated_output):
    #Comparamos lo generado por la red discriminativa con el real
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_generated_output)

    generated_loss = loss_object(tf.zeros_like(disc_real_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

def generator_loss(loss_object, disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss

def set_optimizers():
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    return generator_optimizer, discriminator_optimizer

def set_checkpoints():
    generator_optimizer, discriminator_optimizer = set_optimizers()
    checkpoint_prefix = os.path.join(CKPATH, 'ckpt')
    checkpoint = tf.train.Checkpoint(
        generator_optimizer= generator_optimizer,
        discriminator_optimizer= discriminator_optimizer,
        generator= generator,
        discriminator=discriminator
    )

def restore_checkpoint(checkpoint):
    checkpoint.restore(tf.train.latest_checkpoint(CKPATH)).assert_consumed()

main()