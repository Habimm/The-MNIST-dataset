from info import info
from keras.datasets import mnist
import datetime
import json
import os
import pandas
import tensorflow

class FitLogCallback(tensorflow.keras.callbacks.Callback):

  def __init__(self, fit_log_path):
    tensorflow.keras.callbacks.Callback.__init__(self)
    self.fit_log_file = open(fit_log_path, 'w')
    print('timestamp,accuracy', file=self.fit_log_file)

  def on_train_batch_end(self, batch, logs=None):
    current_timestamp = datetime.datetime.now().timestamp()
    current_accuracy = logs['accuracy']
    print(f'{current_timestamp},{current_accuracy}', file=self.fit_log_file)

# repo_path == '/the_mnist_dataset'
def handle(pb2_request, repo_path):
  # config_json is a <class 'dict'>
  config_json = json.loads(pb2_request.input)

  # Load example MNIST data and pre-process it
  # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  # x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
  # x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

  # # Limit the data to 1000 samples
  # x_train = x_train[:1000]
  # y_train = y_train[:1000]
  # x_test = x_test[:1000]
  # y_test = y_test[:1000]

  (x_train, t_train), (x_test, t_test) = mnist.load_data()
  num_classes=10
  t_train = tensorflow.keras.utils.to_categorical(t_train, num_classes)
  t_test = tensorflow.keras.utils.to_categorical(t_test, num_classes)
  train_size=x_train.shape[0]
  test_size=x_test.shape[0]
  x_train=x_train/255
  x_test=x_test/255
  epoch_number = 5
  learning_rate=0.001
  input_layer_configuration = config_json['config']['layers'][0]
  batch_input_shape = input_layer_configuration['config']['batch_input_shape']
  input_shape = batch_input_shape[1:]
  x_train_shape = [train_size] + input_shape
  x_train = x_train.reshape(x_train_shape)
  x_test_shape = [test_size] + input_shape
  x_test = x_test.reshape(x_test_shape)

  # tensorflow.keras.Model
  model = tensorflow.keras.models.model_from_json(json.dumps(config_json))
  optimiser=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate, name='Adam')
  model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])

  traces_path = os.path.join(f'.{repo_path}', 'traces')

  # Makes a folder for storing the png images
  if not os.path.exists(traces_path):
    os.mkdir(traces_path)

  joined_fit_log_path = os.path.join(f'.{repo_path}', 'traces', 'fit_log.csv')

  training_history = model.fit(
    x_train,
    t_train,
    batch_size=32,
    verbose=0,
    epochs=epoch_number,
    validation_data=(x_test, t_test),
    callbacks=[FitLogCallback(joined_fit_log_path)],
    validation_freq=1,
  )
  # result = model.evaluate(x_test, t_test)

  fit_log_table = pandas.read_csv(joined_fit_log_path)

  # Enrich the JSON for the frontend with more structure
  angefangen = fit_log_table.iloc[0]['timestamp']
  fit_log_table['elapsed_time'] = fit_log_table['timestamp'].map(
    lambda timestamp: f'{timestamp - angefangen:.4f}s'
  )
  fit_log_table['x'] = fit_log_table['timestamp'].map(
    lambda timestamp: timestamp - angefangen
  )
  # ValueError: Invalid value NaN (not a number)
  fit_log_table['datetime'] = fit_log_table['timestamp'].map(
    lambda timestamp:
      datetime.datetime.
      fromtimestamp(timestamp).
      strftime('%H:%M:%S.%f')
  )
  fit_log_table['relative_accuracy'] = fit_log_table['accuracy'].map(
    lambda accuracy: f'{accuracy * 100:.2f}%'
  )
  fit_log_table['y'] = fit_log_table['accuracy']
  # Does the
  # tensorflow.keras.models.model_from_json.fit
  # callbacks
  # tensorflow.keras.callbacks.Callback.on_train_batch_end.logs['accuracy']
  # refer to t_test or t_train?
  fit_log_table['absolute_accuracy'] = (fit_log_table['accuracy'] * len(t_test)).astype('int64')

  fit_log_json = fit_log_table.to_json(orient='records', indent=2)
  json_fit_log_path = os.path.join(f'.{repo_path}', 'traces', 'fit_log.json')
  with open(json_fit_log_path, 'w') as json_path:
    print(fit_log_json, file=json_path)

  return fit_log_json
