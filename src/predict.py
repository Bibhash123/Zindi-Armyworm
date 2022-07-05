
import os
import glob
import random
from PIL import Image
import cv2
import numpy as np
import requests
import pandas as pd
import zipfile
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tqdm import tqdm
import tensorflow_hub as hub

def seedAll(seed):
  np.random.seed(seed)
  random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  tf.random.set_seed(seed)

seedAll(101)

url = "https://api.zindi.africa/v1/competitions/makerere-fall-armyworm-crop-challenge/files/"
token = {'auth_token': '56eo46oXnWGoHHmrLLDq5xRJ'} 
def zindi_data_downloader(url, token, file_name):
    # Get the competition data
    competition_data = requests.post(url = f"{url}{file_name}", data= token, stream=True)
    
    # Progress bar monitor download
    pbar = tqdm(desc=file_name, total=int(competition_data.headers.get('content-length', 0)), unit='B', unit_scale=True, unit_divisor=512)
    # Create and Write the data to colab drive in chunks
    handle = open(file_name, "wb")
    for chunk in competition_data.iter_content(chunk_size=512): # Download the data in chunks
        if chunk: # filter out keep-alive new chunks
                handle.write(chunk)
        pbar.update(len(chunk))
    handle.close()
    pbar.close()

print("[INFO] Downloading Data")

zindi_data_downloader(url,token,"Train.csv")
zindi_data_downloader(url,token,"Test.csv")
zindi_data_downloader(url,token,"SampleSubmission.csv")
zindi_data_downloader(url,token,"Images.zip")
zipfile.ZipFile('Images.zip','r').extractall("./data")

IMG_SIZE = 224
BATCH_SIZE = 16
AUTOTUNE = tf.data.experimental.AUTOTUNE

test = pd.read_csv("./Test.csv")

from functools import partial
def decode_fn(is_labelled):
  def without_labels(path, segpath, target_size=IMG_SIZE, test = False):
    file_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(file_bytes, channels = 0)
    if img.shape[0]!=target_size:
      img = tf.image.resize(img, (target_size,target_size))
    
    file_bytes = tf.io.read_file(path)
    segimg = tf.image.decode_jpeg(file_bytes, channels = 0)
    if segimg.shape[0]!=target_size:
      segimg = tf.image.resize(segimg, (target_size,target_size))
    # img = centerCrop(img,100)
    if not test:
      return img,segimg
    else:
      return (img,segimg),tf.ones((1,1))
  
  def with_label(path, segimg, label, target_size=IMG_SIZE):
    return without_labels(path, segimg, target_size), label

  return with_label if is_labelled else without_labels

def createTFDatasetSegment(dataframe,
                    batch_size,
                    batch = True,
                    is_labelled = True,
                    shuffle = True,
                    cache = True,
                    repeat = True,
                    file_path = "/content/data/",
                    seg_path = "/content/segmented/images/",
                    test=False):
  df = dataframe.copy()
  df['paths'] = df['Image_id'].apply(lambda x: os.path.join(file_path,x))
  df['segpaths'] = df['Image_id'].apply(lambda x: os.path.join(seg_path,x))

  decoder = decode_fn(is_labelled)
  if is_labelled:
    dataset = tf.data.Dataset.from_tensor_slices((df['paths'].values,
                                                  df['segpaths'].values,
                                                  df['Label'].values
                                                  ))
  else:
    dataset = tf.data.Dataset.from_tensor_slices((df['paths'].values,
                                                  df['segpaths'].values))
  dataset = dataset.map(partial(decoder,test = test), num_parallel_calls = AUTOTUNE)
  dataset = dataset.cache("") if cache else dataset
  dataset = dataset.repeat() if repeat else dataset
  dataset = dataset.shuffle(1024, reshuffle_each_iteration = True) if shuffle else dataset
  dataset = dataset.batch(batch_size,drop_remainder=False) if batch else dataset
  dataset = dataset.prefetch(AUTOTUNE)
  return dataset

def buildSegmentModel():
  m1 = tf.keras.applications.efficientnet.EfficientNetB0(include_top = False, weights = 'imagenet', input_shape= (IMG_SIZE,IMG_SIZE,3))
  total = len(m1.layers)
  for idx,layer in enumerate(m1.layers):
    if idx>total-30:
      layer.trainable = True
    else:
      layer.trainable = False

  m2 = tf.keras.applications.efficientnet.EfficientNetB0(include_top = False, weights = 'imagenet', input_shape= (IMG_SIZE,IMG_SIZE,3))
  total = len(m2.layers)
  for idx,layer in enumerate(m2.layers):
    if idx>total-30:
      layer.trainable = True
    else:
      layer.trainable = False

  augmenter = tf.keras.Sequential([
                                   L.RandomFlip(mode="horizontal_and_vertical"),
                                   L.RandomRotation(0.1),
                                   L.RandomContrast(0.1)
  ])

  head1 = tf.keras.Sequential([
                      m1,
                      L.Flatten()
                    ])
  
  head2 = tf.keras.Sequential([
                      m2,
                      L.Flatten()
                    ])
  
  inp = L.Input(shape = (IMG_SIZE,IMG_SIZE,3))
  inp2 = L.Input(shape = (IMG_SIZE,IMG_SIZE,3))
  # aug = augmenter(inp)
  h1 = head1(inp)
  h2 = head2(inp2)
  h = L.Concatenate(axis=-1)([h1,h2])
  h = L.Dropout(0.1)(h)
  h = L.Dense(128,activation='relu')(h)
  h = L.BatchNormalization()(h)
  h = L.Dropout(0.1)(h)
  h = L.Dense(64,activation='relu')(h)
  h = L.BatchNormalization()(h)
  h = L.Dropout(0.1)(h)
  h = L.Dense(32,activation='relu')(h)
  h = L.BatchNormalization()(h)
  h = L.Dropout(0.1)(h)
  h = L.Dense(16,activation='relu')(h)
  h = L.Dropout(0.1)(h)
  h = L.Dense(1,activation='sigmoid')(h)

  model = tf.keras.Model(inputs = [inp,inp2],outputs = h)
  schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-3,
                                                            decay_steps=140,
                                                            decay_rate=0.96,
                                                            staircase=True)  
  optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
  model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ["accuracy"])
  return model

def buildModel():
  m1 = tf.keras.applications.efficientnet.EfficientNetB0(include_top = False, weights = 'imagenet', input_shape= (IMG_SIZE,IMG_SIZE,3))
  total = len(m1.layers)
  for idx,layer in enumerate(m1.layers):
    if idx>total-30:
      layer.trainable = True
    else:
      layer.trainable = False

  augmenter = tf.keras.Sequential([
                                   L.RandomFlip(mode="horizontal_and_vertical"),
                                   L.RandomRotation(0.1),
                                   L.RandomContrast(0.1)
  ])

  head1 = tf.keras.Sequential([
                      m1,
                      L.Flatten()
                    ])
  
  inp = L.Input(shape = (IMG_SIZE,IMG_SIZE,3))
  # aug = augmenter(inp)
  h = head1(inp)
  h = L.Dropout(0.1)(h)
  h = L.Dense(128,activation='relu')(h)
  h = L.BatchNormalization()(h)
  h = L.Dropout(0.1)(h)
  h = L.Dense(64,activation='relu')(h)
  h = L.BatchNormalization()(h)
  h = L.Dropout(0.1)(h)
  h = L.Dense(32,activation='relu')(h)
  h = L.BatchNormalization()(h)
  h = L.Dropout(0.1)(h)
  h = L.Dense(16,activation='relu')(h)
  h = L.Dropout(0.1)(h)
  h = L.Dense(1,activation='sigmoid')(h)

  model = tf.keras.Model(inputs = inp,outputs = h)
  schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-3,
                                                            decay_steps=140,
                                                            decay_rate=0.96,
                                                            staircase=True)  
  optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
  model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ["accuracy"])
  return model

def decode_fn_(is_labelled):
  def without_labels(path, target_size=IMG_SIZE):
    file_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(file_bytes, channels = 0)
    if img.shape[0]!=target_size:
      img = tf.image.resize(img, (target_size,target_size))
    # img = centerCrop(img,100)
    return img
  
  def with_label(path, label, target_size=IMG_SIZE):
    return without_labels(path, target_size), label

  return with_label if is_labelled else without_labels

def createTFDataset(dataframe,
                    batch_size,
                    batch = True,
                    is_labelled = True,
                    shuffle = True,
                    cache = True,
                    repeat = True,
                    file_path = "/content/data/"):
  df = dataframe.copy()
  df['paths'] = df['Image_id'].apply(lambda x: os.path.join(file_path,x))

  decoder = decode_fn_(is_labelled)
  if is_labelled:
    dataset = tf.data.Dataset.from_tensor_slices((df['paths'].values,
                                                  df['Label'].values
                                                  ))
  else:
    dataset = tf.data.Dataset.from_tensor_slices((df['paths'].values))
  dataset = dataset.map(decoder, num_parallel_calls = AUTOTUNE)
  dataset = dataset.cache("") if cache else dataset
  dataset = dataset.repeat() if repeat else dataset
  dataset = dataset.shuffle(1024, reshuffle_each_iteration = True) if shuffle else dataset
  dataset = dataset.batch(batch_size,drop_remainder=False) if batch else dataset
  dataset = dataset.prefetch(AUTOTUNE)
  return dataset


tst_segment = createTFDatasetSegment(test,
                      batch_size = BATCH_SIZE,
                      batch = True,
                      is_labelled = False,
                      shuffle = False,
                      cache = False,
                      repeat = False,
                      file_path = "./data/",
                      seg_path = "../input/Segmented_Images/",
                      test = True)

tst = createTFDatasetSegment(test,
                      batch_size = BATCH_SIZE,
                      batch = True,
                      is_labelled = False,
                      shuffle = False,
                      cache = False,
                      repeat = False,
                      file_path = "./data/")

pred = np.zeros((test.shape[0],1))

print("[INFO] Predicting for Base Model")
for fold in tqdm(range(5)):
  model = buildModel()
  model.load_weights(f'../models/base/model_{fold}.hdf5')
  pred += model.predict(tst)/10

print("[INFO] Predicting for Segmentation Model")
for fold in tqdm(range(5)):
  model = buildSegmentModel()
  model.load_weights(f'../models/segmented/model_{fold}.hdf5')
  pred += model.predict(tst_segment)/10


ss = pd.read_csv("./SampleSubmission.csv")
ss.iloc[:,1] = pred

ss.to_csv("./submission.csv", index = False)
print("[INFO] Predictions written to file submission.csv")