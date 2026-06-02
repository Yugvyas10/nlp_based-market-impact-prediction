import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns

training_set = tf.keras.utils.image_dataset_from_directory(
    '/Users/yugvyas/Documents/NLP-based_market_predictions/Input_Data/train-00000-of-00001.parquet',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    
)