from init import build_model
from train import train_model
from test import test_model
from dataset import read_scale_dataset
import tensorflow as tf

# Configuration de la mémoire GPU
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Lecture des données
dataset = read_scale_dataset()

# Initialisation du modèle
model = build_model()

# Entraînement du modèle
train_model(model, [dataset[0], dataset[1]])

# Test du modèle
test_model(model, [dataset[2], dataset[3]])

# Save
model.save('model.h5')
