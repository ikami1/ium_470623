import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow as tf
import math

from tensorflow import keras
from process_dataset import process_data_and_get_x_y


def show_result(x, y):
    plt.title('Usage kWh Model', fontsize=15, color='g', pad=12)
    plt.plot(x, y, 'o', color='r')

    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b, color='darkblue')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()


model = keras.models.load_model('steel_industry_model')

energy_data_test = pd.read_csv('steel_industry_data_test.csv')
energy_data_test, x_test, y_test = process_data_and_get_x_y(energy_data_test)

y_predicted = model.predict(x_test)
test_results = {}
test_results['usage_model'] = model.evaluate(
    x_test,
    y_test, verbose=0)

print('Mean Absolute Error : ', metrics.mean_absolute_error(y_test, y_predicted))
print('Mean Squared Error : ', metrics.mean_squared_error(y_test, y_predicted))
print('Root Mean Squared Error : ', math.sqrt(metrics.mean_squared_error(y_test, y_predicted)))

print(test_results['usage_model'])

#show_result(y_test, y_predicted)

with open('eval_results.txt', 'a+') as f:
    f.write(str(test_results['usage_model']) + '\n')
