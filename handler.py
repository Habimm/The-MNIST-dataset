from info import info
from keras.datasets import mnist
import json
import tensorflow

def handle(request):
  info(request)
  info(request.headers)
  content_type = request.headers.get('Content-Type')
  if (content_type == 'application/json'):
    config_json = request.json
  else:
    return f'Content-Type {content_type} not supported!'

  # load data
  # split data into training and test
  (x_train, t_train), (x_test, t_test) = mnist.load_data()
  num_classes=10
  # One hot vectors
  t_train = tensorflow.keras.utils.to_categorical(t_train, num_classes)
  t_test = tensorflow.keras.utils.to_categorical(t_test, num_classes)
  train_size=x_train.shape[0]
  num_features=x_train.shape[1]
  test_size=x_test.shape[0]
  x_train=x_train/255
  x_test=x_test/255
  epoch_number = 5
  # x_train has changed from the above execution of the same command
  num_features=x_train.shape[1]
  learning_rate=0.1

  input_layer_configuration = config_json['config']['layers'][0]
  batch_input_shape = input_layer_configuration['config']['batch_input_shape']
  input_shape = batch_input_shape[1:]
  x_train_shape = [train_size] + input_shape
  x_train = x_train.reshape(x_train_shape)
  x_test_shape = [test_size] + input_shape
  x_test = x_test.reshape(x_test_shape)

  # Returns A Keras model instance (uncompiled).
  # https://www.tensorflow.org/api_docs/python/tf/keras/models/model_from_json
  # config = model.to_json()
  # We are awaiting a str, not a dict: TypeError: the JSON object must be str, bytes or bytearray, not dict
  model = tensorflow.keras.models.model_from_json(json.dumps(config_json))
  # model_json = model.to_json()
  optimiser=tensorflow.keras.optimizers.Adam(learning_rate=0.001, name='Adam')
  model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])

  training_history = model.fit(
    x_train, # input
    t_train, # output
    batch_size=32,
    verbose=0, # Suppress chatty output; use Tensorboard instead
    epochs=epoch_number,
    validation_data=(x_test, t_test),
  )
  result = model.evaluate(x_test, t_test)

  stringlist = []
  model.summary(print_fn=lambda x: stringlist.append(x))
  short_model_summary = "\n".join(stringlist)
  info(short_model_summary)

  response_output_dictionary = {
    # 'input': config_json,
    # 'summary': short_model_summary,
    # 'epoch_number': epoch_number,
    'output': result,
  }

  return json.dumps(response_output_dictionary, indent=2)
