from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Activation, UpSampling2D, Conv2D
from tensorflow_examples.models.pix2pix import pix2pix

from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import os
import time
import matplotlib.pyplot as plt
import struct
import numpy
import sys
import librosa
import scipy
from matplotlib import *
from librosa import display
from IPython.display import clear_output
from datetime import datetime
tfds.disable_progress_bar()
AUTOTUNE = tf.data.experimental.AUTOTUNE

#tf.keras.backend.set_floatx('float64')
tf.keras.backend.set_floatx('float32')
#numpy.set_printoptions(threshold=sys.maxsize)


numpy.seterr(divide='ignore', invalid='ignore')





numInst1 = 160
numInst2 = 139 #Number of files in the folder. I'll upgrade it when I have to.



#########################
LAMBDA = 1
EPOCHS = 20
TRAINING_RATIO = 1
LEARNING_RATE = 2e-4 # 2e-4
IMG_WIDTH=256
IMG_HEIGHT=256
BUFFER_SIZE = 139
SAMPLE_RATE = 16000
TRAIN_SIZE = 100

FILTER_SCALE = 1
OUTPUT_CHANNELS = 1

CQT_WIDTH = 256
CQT_HEIGHT = 126
########################

def random_crop(image):
  cropped_image = tf.image.random_crop(image, size=[IMG_WIDTH,IMG_HEIGHT])
  return cropped_image



  # normalizing the images to [-1, 1]

def random_jitter(image):
 # image = tf.image.resize(image, [286, 286],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) # resizing to 286 x 286 x 3
  image = numpy.resize(image, (IMG_WIDTH+28,IMG_HEIGHT+28)) # resizing to 286 x 286 x 3
 
  image = random_crop(image)  # randomly cropping to 256 x 256 x 3
  
  # image = tf.image.random_flip_left_right(image)  # Mirroring would take information away from the spectrogram.... I..... think?
  return image



def discriminator_loss(real, generated):
  real_loss = loss_object(tf.ones_like(real), real)
  generated_loss = loss_object(tf.zeros_like(generated), generated)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss * 0.5
def generator_loss(generated):
  return loss_object(tf.ones_like(generated), generated)
def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  return LAMBDA * loss1
def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

####
d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
####






import datetime




#tensorboard
# Define our metrics
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_gen_loss = tf.keras.metrics.Mean('gen_loss', dtype=tf.float32)
train_disc_loss = tf.keras.metrics.Mean('disc_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)




@tf.function
def train_step(real_x, real_y,epoch):
    
    
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.

    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)



    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)

    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
    
    tf.print("Total Cycle Loss: ")
    tf.print(total_cycle_loss)
    tf.print("Generator Loss: ")
    tf.print(gen_g_loss)
    tf.print("Discriminator Loss: ")
    tf.print(disc_x_loss)
    
    

    
    
    ##################################################################
    #tensorboard operations
    train_loss(total_cycle_loss)
    train_gen_loss(gen_g_loss)
    train_disc_loss(disc_x_loss)
    train_accuracy(cycled_x,real_x)
    
    


    with train_summary_writer.as_default():
        tf.summary.scalar('Cycle Loss', train_loss.result(), step=epoch)
        tf.summary.scalar('Generator Loss', train_gen_loss.result(), step=epoch)
        tf.summary.scalar('Discriminator Loss', train_disc_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        

    with test_summary_writer.as_default():
        tf.summary.scalar('Cycle Loss', test_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)



    ##################################################################
    
      # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss,
                                        generator_g.trainable_variables)
  
  generator_f_gradients = tape.gradient(total_gen_f_loss,
                                        generator_f.trainable_variables)

  discriminator_x_gradients = tape.gradient(disc_x_loss,
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss,
                                            discriminator_y.trainable_variables)

  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                            generator_f.trainable_variables))

  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))

  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))

######################################################################

def display_spectrogram(spec,title):
  librosa.display.specshow(librosa.amplitude_to_db(spec, ref=numpy.max),sr=SAMPLE_RATE, x_axis='time', y_axis='cqt_note')
  plt.colorbar(format='%+2.0f dB')
  plt.title(title) 
  plt.tight_layout()
  plt.show()

def save_spectrogram(spec,title,count):
  librosa.display.specshow(librosa.amplitude_to_db(spec, ref=numpy.max),sr=SAMPLE_RATE, x_axis='time', y_axis='cqt_note')
  plt.axis('off')
  
  frame1 = plt.gca()
  frame1.axis('off')
  frame1.axes.xaxis.set_ticklabels([])
  frame1.axes.yaxis.set_ticklabels([])
  frame1.axes.xaxis.label.set_visible(False)
  frame1.axes.yaxis.label.set_visible(False)
  
  
  
  plt.savefig("SpectrogramImages/"+title+str(count)+'.png',bbox_inches='tight', pad_inches=0)
  plt.close()


#def discriminator_loss(disc_real_output, disc_generated_output):
#  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

#  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

#  total_disc_loss = real_loss + generated_loss

#  return total_disc_loss

def generate_images(model, test_input,count,tinst1max):

  

  #prediction = tf.image.rgb_to_grayscale(model(test_input))
  #nti = (numpy.reshape(tf.image.rgb_to_grayscale(test_input),(1,IMG_WIDTH,IMG_HEIGHT))[0])
  
  prediction = model(test_input)
  prediction = numpy.reshape(prediction,(1,IMG_WIDTH,IMG_HEIGHT))[0]
  nti = numpy.reshape(test_input,(1,IMG_WIDTH,IMG_HEIGHT))[0]
  
  
  nnti = nti#[0:CQT_WIDTH, 0:CQT_HEIGHT]
  nprediction = prediction#[0:CQT_WIDTH,0:CQT_HEIGHT]
 
  #nprediction = (nprediction + 1)/2
  #nnti = (nnti + 1)/2
  
  #nprediction = (nprediction*2 - 1) * 0.0001
  #nprediction = normalise_specs(nprediction)
  #nnti = numpy.exp(nnti)# * 0.0001
  
  #nprediction = numpy.exp(nprediction) * 0.000005/1.2
  #nnti = numpy.exp(nnti) +5
  
  #nprediction = numpy.exp(nprediction) # * 0.0001
  #nprediction = 1 - nprediction
  #nnti = ((nnti + 1)/2) * 4.925224863246271
  nnti = (nnti + 1) / 2
  #print(nnti)
  
  nprediction = (nprediction + 1)/2
  #print(nprediction)
  
  nnti = nnti * tinst1max
  nprediction = nprediction * tinst1max
  
  nnti = numpy.exp(nnti) - 1
  nprediction = numpy.exp(nprediction)-1
  
  #nprediction = numpy.exp((nprediction+1)/2)
  #nprediction = numpy.exp(nprediction)

  #print(nnti)
  #print(nprediction)
  
  
  nnti = nnti[0:CQT_WIDTH, 0:CQT_HEIGHT]
  nprediction = nprediction[0:CQT_WIDTH,0:CQT_HEIGHT]

  
  save_spectrogram(nnti,"Test",count)
  save_spectrogram(nprediction,"Prediction",count)
  

  
  #hola = librosa.core.griffinlim_cqt(newnum,sr=16000,hop_length=128,dtype = numpy.float64,bins_per_octave = int(12)) #using this one atm
  #hola = librosa.core.icqt(gg,sr=16000,hop_length=128,bins_per_octave = 2*36,filter_scale=0.1)
  
  
  
  ############################################################################################################################################################
  #hola = librosa.icqt(gg,sr=SAMPLE_RATE,hop_length=64,bins_per_octave = 4*36,filter_scale=1)

  #hola = librosa.griffinlim(nprediction,hop_length = 128)
  #hola2 = librosa.griffinlim(nnti,hop_length = 128)
  hola = librosa.feature.inverse.mel_to_audio(M = nprediction, sr=SAMPLE_RATE)
  hola2 = librosa.feature.inverse.mel_to_audio(M = nnti, sr=SAMPLE_RATE)
  
  hola=librosa.util.normalize(hola)
  hola2=librosa.util.normalize(hola2)
  
  librosa.output.write_wav("audio_prediction"+str(count)+".wav",hola,16000)
  librosa.output.write_wav("audio_test"+str(count)+".wav",hola2,16000)#, hop_length=1024, fmin=300, bins_per_octave=int(10*4))

  ############################################################################################################################################################

###



    

def normalise_specs(x):
    """
    x = list of spectrograms
    """
    norm_x=[]
    for i in range(len(x)):
        xmax, xmin = x[i].max(), x[i].min()
        norm_x.append(numpy.nan_to_num((x[i] - xmin)/(xmax - xmin)))
    return numpy.array(norm_x)   

    

def lognormalise_specs(x):
    lognorm_x = numpy.log(x + 1)
    return lognorm_x


def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {"label": tf.compat.v1.FixedLenFeature([], tf.int64),'image': tf.compat.v1.FixedLenFeature([], tf.string)}
    # Load one example
    parsed_features = tf.compat.v1.parse_single_example(proto, keys_to_features)
    # Turn your saved image string into an array
    parsed_features['image'] = tf.compat.v1.decode_raw(parsed_features['image'], tf.float64)
    return parsed_features['image']#, parsed_features["label"]

inst1_arr = []
train_inst1s = tf.compat.v1.data.TFRecordDataset("trainA.tfrecord")
train_inst1s = train_inst1s.map(_parse_function, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)   
for image in train_inst1s:
    imgarray = numpy.array(image)
    image = image.numpy()
    inst1_arr.append(image)
    
    
inst2_arr = []
train_inst2s = tf.compat.v1.data.TFRecordDataset("trainB.tfrecord")
train_inst2s = train_inst2s.map(_parse_function, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)    
for image in train_inst2s:
    imgarray = numpy.array(image)
    image = image.numpy()
    inst2_arr.append(image)
    
test_arr = []
test_inst1s = tf.compat.v1.data.TFRecordDataset("testA.tfrecord")
test_inst1s = test_inst1s.map(_parse_function, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)   
for image in test_inst1s:
    imgarray = numpy.array(image)
    image = image.numpy()
    test_arr.append(image)
    
    


inst2_np = numpy.asarray(inst2_arr)
inst1_np = numpy.asarray(inst1_arr)
test_np = numpy.asarray(test_arr)



shapedinst2_np=numpy.asarray(numpy.abs(inst2_np.reshape(numInst1,IMG_WIDTH,IMG_HEIGHT))) # 139 
shapedinst1_np=numpy.asarray(numpy.abs(inst1_np.reshape(numInst2,IMG_WIDTH,IMG_HEIGHT))) # 160

test_np=numpy.asarray(numpy.abs(test_np.reshape(10,IMG_WIDTH,IMG_HEIGHT)))



###Generator Functions

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(tf.keras.layers.Conv2D(filters, size,
                             strides=2,padding='same',
                             kernel_initializer=initializer, use_bias=False))
  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.LeakyReLU())
  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
  result.add(tf.keras.layers.BatchNormalization())
  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))
  result.add(tf.keras.layers.ReLU())
  return result

def Generator(NUM_CHANNELS):
  inputs = tf.keras.layers.Input(shape=[256,256,NUM_CHANNELS])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):#
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


#def generator_loss(disc_generated_output, gen_output, target):
#  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
  # mean absolute error
#  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
#  total_gen_loss = gan_loss + (LAMBDA * l1_loss)
#  return total_gen_loss, gan_loss, l1_loss


####################################################################################
##Discriminator Functions

def wasserstein_loss(y_true, y_pred):
	return mean(y_true * y_pred)

def Discriminator(NUM_CHANNELS,norm_type='batchnorm', target=False):
  """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
  Args:
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    target: Bool, indicating whether target image is an input or not.
  Returns:
    Discriminator model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, NUM_CHANNELS], name='input_image')
  x = inp

  if target:
    tar = tf.keras.layers.Input(shape=[256, 256, NUM_CHANNELS], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, norm_type)(x)  # (bs, 128, 128, 64)
  down2 = downsample(128, 4, norm_type)(down1)  # (bs, 64, 64, 128)
  down3 = downsample(256, 4, norm_type)(down2)  # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer,use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

  if norm_type.lower() == 'batchnorm':
    norm1 = tf.keras.layers.BatchNormalization()(conv)
  elif norm_type.lower() == 'instancenorm':
    norm1 = InstanceNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

  if target:
    return tf.keras.Model(inputs=[inp, tar], outputs=last)
  else:
    return tf.keras.Model(inputs=inp, outputs=last)



#def discriminator_loss(disc_real_output, disc_generated_output):
#  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

#  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

#  total_disc_loss = real_loss + generated_loss

#  return total_disc_loss


#######################
##Generator-Discriminator Instantiation #These are the built in pix-2-pix models that were used as placeholders
#generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
#generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
#discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
#discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

##These are custom written models
generator_g = Generator(1);
generator_f = Generator(1);
discriminator_x = Discriminator(1);
discriminator_y = Discriminator(1);


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#print("Objective Loss: " + str(loss_obj))



###################################################################
# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
	# plot loss
	pyplot.subplot(2, 1, 1)
	pyplot.plot(d1_hist, label='d-real')
	pyplot.plot(d2_hist, label='d-fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	# plot discriminator accuracy
	pyplot.subplot(2, 1, 2)
	pyplot.plot(a1_hist, label='acc-real')
	pyplot.plot(a2_hist, label='acc-fake')
	pyplot.legend()
	# save plot to file
	pyplot.savefig('results_baseline/plot_line_plot_loss.png')
	pyplot.close()
###################################################################


generator_g_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)
discriminator_x_optimizer = tf.keras.optimizers.Adam((LEARNING_RATE)/TRAINING_RATIO, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam((LEARNING_RATE)/TRAINING_RATIO, beta_1=0.5)

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,  #Loading in checkpoint parameters
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')



for epoch in range(EPOCHS):
  start = time.time()
  n = 0
  for i in range(0,TRAIN_SIZE):
    

    tinst1_np = numpy.array(shapedinst1_np[i])
    tinst2_np = numpy.array(shapedinst2_np[i])
    
    tinst1_np = (lognormalise_specs(tinst1_np))
    tinst1max = numpy.max(tinst1_np)
    tinst1_np = tinst1_np/tinst1max
    
    tinst2_np = (lognormalise_specs(tinst2_np))
    tinst2max = numpy.max(tinst2_np)
    tinst2_np = tinst2_np/tinst2max
    
    tinst1_np = tinst1_np * 2 - 1
    tinst2_np = tinst2_np * 2 - 1
    
    #tinst1_np = librosa.util.normalize(tinst1_np)
    #tinst2_np = librosa.util.normalize(tinst2_np)
    
    
    
    ###Test does not need random jitter
    tinst1_np = random_jitter(tinst1_np)
    tinst2_np = random_jitter(tinst2_np) 

    tinst2_np = numpy.stack((tinst2_np,)*1, axis=-1) #no need because we modified the code
    tinst1_np = numpy.stack((tinst1_np,)*1, axis=-1)
    
    tinst2_np = numpy.stack((tinst2_np,)*1, axis=0) #no need because we modified the code
    tinst1_np = numpy.stack((tinst1_np,)*1, axis=0)
    
 #   tinst2_np = tinst2_np.reshape((1,) + tinst2_np.shape) 
 #   tinst1_np = tinst1_np.reshape((1,) + tinst1_np.shape) 
    
    tinst2_np = tf.cast(tf.convert_to_tensor(tinst2_np),tf.float32)
    tinst1_np = tf.cast(tf.convert_to_tensor(tinst1_np),tf.float32)

    
    



    train_step(tinst1_np, tinst2_np,epoch)
  clear_output(wait=True)


  if (epoch + 1) % 10 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))
  
  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))


test_count = 0
for inp in range(10):
    tinst1_np = test_np[inp]
    

    
    
    tinst1_np = (lognormalise_specs(tinst1_np))
    tinst1max = numpy.max(tinst1_np)
    tinst1_np = tinst1_np/tinst1max
    
    
    
    tinst1_np = tinst1_np * 2 - 1
    
    print(tinst1_np)



######################################################
    tinst1_np = numpy.stack((tinst1_np,)*1, axis=-1)
    
    tinst1_np = numpy.stack((tinst1_np,)*1, axis=0)
#######################################################




    #Pre-processing
 #   tinst1_np = numpy.stack((tinst1_np,)* 3, axis=-1)  #Don't have to do this because the code deals with one channel only
#    tinst1_np = tinst1_np.reshape((1,) + tinst1_np.shape)     
 #   tinst1_np = tf.convert_to_tensor(tinst1_np)
    tinst1_np = tf.cast(tf.convert_to_tensor(tinst1_np),tf.float32)
    
    
    generate_images(generator_g, tinst1_np,test_count,tinst1max)
    
    test_count = test_count + 1
    
    
plt.show()

