import tensorflow as tf
import random
import os
import numpy as np

try:
  from os import scandir
except ImportError:
  # Python 2 polyfill module
  from scandir import scandir
    

FLAGS = tf.compat.v1.flags.FLAGS


##this script processes files in the output directory

tf.compat.v1.flags.DEFINE_string('X_input_dir', 'output/TrainA',
                       'X input directory, default: output/TrainA')
tf.compat.v1.flags.DEFINE_string('Y_input_dir', 'output/TrainB',
                       'Y input directory, default: output/TrainB')
tf.compat.v1.flags.DEFINE_string('X_output_file', 'output/tfrecords/trainA.tfrecord',
                       'X output tfrecords file, default: output/tfrecords/trainA.tfrecord')
tf.compat.v1.flags.DEFINE_string('Y_output_file', 'output/tfrecords/trainB.tfrecord',
                       'Y output tfrecords file, default: output/tfrecords/trainB.tfrecord')

tf.compat.v1.flags.DEFINE_string('X_input_dir_test', 'output/TestA',
                       'X test input directory, default: output/TestA')
tf.compat.v1.flags.DEFINE_string('Y_input_dir_test', 'output/TestB',
                       'Y test input directory, default: output/TestB')
tf.compat.v1.flags.DEFINE_string('X_output_file_test', 'output/tfrecords/testA.tfrecord',
                       'X test output tfrecords file, default: output/tfrecords/testA.tfrecord')
tf.compat.v1.flags.DEFINE_string('Y_output_file_test', 'output/tfrecords/testB.tfrecord',
                       'Y test output tfrecords file, default: output/tfrecords/testB.tfrecord')



def data_reader(input_dir, shuffle=True):
  """Read images from input_dir then shuffle them
  Args:
    input_dir: string, path of input dir, e.g., /path/to/dir
  Returns:
    file_paths: list of strings
  """
  file_paths = []

  for img_file in scandir(input_dir):
    if img_file.name.endswith('.npy') and img_file.is_file():
      file_paths.append(img_file.path)

  if shuffle:
    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable with seed value.
    shuffled_index = list(range(len(file_paths)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    file_paths = [file_paths[i] for i in shuffled_index]

  return file_paths


def _int64_feature(value):
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=n_value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(file_path, image_buffer):
  """Build an Example proto for an example.
  Args:
    file_path: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
  Returns:
    Example proto
  """
  file_name = file_path.split('/')[-1]

  print(image_buffer)
  print(np.shape(image_buffer))

  example = tf.train.Example(features=tf.train.Features(feature={
      'label': _int64_feature(0),
    #  'image': _bytes_feature((image_buffer))
      'image': _bytes_feature(tf.compat.as_bytes(image_buffer.tostring()))
      
    }))
  return example









def data_writer(input_dir, output_file):
  """Write data to tfrecords
  """
  file_paths = data_reader(input_dir)

  # create tfrecords dir if not exists
  output_dir = os.path.dirname(output_file)
  try:
    os.makedirs(output_dir)
  except os.error as e:
    pass

  images_num = len(file_paths)

  # dump to tfrecords file
  writer = tf.compat.v1.python_io.TFRecordWriter(output_file)

  for i in range(len(file_paths)):
    file_path = file_paths[i]

    #with tf.compat.v1.gfile.FastGFile(file_path, 'rb') as f:
      #image_data = f.read() #for actual images
      
    image_data = np.load(file_path)#for numpy arrays

    example = _convert_to_example(file_path, image_data)
    writer.write(example.SerializeToString())

    if i % 500 == 0:
      print("Processed {}/{}.".format(i, images_num))
  print("Done.")
  writer.close()














def main(unused_argv):
  print("Convert X Training data to tfrecords...")
  data_writer(FLAGS.X_input_dir, FLAGS.X_output_file)
  print("Convert Y Training data to tfrecords...")
  data_writer(FLAGS.Y_input_dir, FLAGS.Y_output_file)
  
  print("Convert X Test data to tfrecords...")
  data_writer(FLAGS.X_input_dir_test, FLAGS.X_output_file_test)
  print("Convert Y Test data to tfrecords...")
  data_writer(FLAGS.Y_input_dir_test, FLAGS.Y_output_file_test)

if __name__ == '__main__':
  tf.compat.v1.app.run()
