import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow as tf
import math

from tensorflow import keras
#from tensorflow.keras import layers


def process_data_and_get_x_y(data):
    data.columns = ["date", "Usage_kWh", "Lagging_Current_Reactive.Power_kVarh", "Leading_Current_Reactive_Power_kVarh",
                    "CO2(tCO2)", "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor", "WeekStatus",
                    "Day_of_week", "Load_Type"] #without NSM column
    data = data.set_index('date')
    data = pd.get_dummies(data, drop_first=True)

    x = data.drop('Usage_kWh', axis=1)
    #x = data['Lagging_Current_Reactive.Power_kVarh']
    y = data['Usage_kWh']
    return data, x, y


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


energy_data_train = pd.read_csv('Steel_industry_data_train.csv')

energy_data_train, x_train, y_train = process_data_and_get_x_y(energy_data_train)

#x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)
# x_test, x_dev, y_test, y_dev = train_test_split(x_test, y_test, test_size=0.5, random_state=1)

# stats
print(x_train.describe(include='all'))
#print(np.array(x_train).reshape(-1, 1))

normalizer = tf.keras.layers.Normalization(axis=1)
normalizer.adapt(np.array(x_train))
print(normalizer.mean.numpy())

# powinno być niezmienione
print(np.array(x_train[:1]))

usage_model = tf.keras.Sequential([
    normalizer,
    keras.layers.Dense(units=10, activation='relu'),
    keras.layers.Dense(units=1)
])

print(usage_model.summary())

usage_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = usage_model.fit(
    x_train,
    y_train,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split=0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

#plot_loss(history)

usage_model.save('steel_industry_model')
