import tensorflow as tf
import numpy as np
import os
import collections
import random
import time
import matplotlib.pyplot as plt
from PIL import Image
import json

EPOCHS = 30
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embed_size = 256
hidden_size = 512
vocabulary_size = 5001
features_shape = 2048
attention_features_shape = 64

captions_path = '/captions/'
if not os.path.exists(os.path.abspath('.') + captions_path):
  captions_zip = tf.keras.utils.get_file('captions.zip', cache_subdir=os.path.abspath('.'), origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip', extract=True)
  captions_file = os.path.dirname(captions_zip)+'/captions/captions_train2014.json'
  os.remove(captions_zip)

images_path = '/train2014/'
if not os.path.exists(os.path.abspath('.') + images_path):
  images_zip = tf.keras.utils.get_file('train2014.zip', cache_subdir=os.path.abspath('.'), origin='http://images.cocodataset.org/zips/train2014.zip', extract=True)
  PATH = os.path.dirname(images_zip) + images_path
  os.remove(images_zip)
else:
  PATH = os.path.abspath('.') + images_path

with open(captions_file, 'r') as f:
    captions = json.load(f)

def load_image(path_image):
    image = tf.io.read_file(path_image)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (299, 299))
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image, path_image

# Calculation of  highest of all caption lenghts
def caption_length_max(x):
    return max(len(t) for t in x)

def map_func(image_name, cap):
  image_tensor = np.load(image_name.decode('utf-8')+'.npy')
  return image_tensor, cap

def calc_loss(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  new_loss = new_lossobject(real, pred)
  mask = tf.cast(mask, dtype=new_loss.dtype)
  new_loss *= mask

  return tf.reduce_mean(new_loss)

captions_of_image = collections.defaultdict(list)
for cap in captions['captions']:
  caption = f"<start> {cap['caption']} <end>"
  path_image = PATH + 'COCO_train2014_' + '%012d.jpg' % (cap['image_id'])
  captions_of_image[path_image].append(caption)

path_images = list(captions_of_image.keys())
random.shuffle(path_images)

path_train = path_images[:6000]
captions_list = []
image_list = []

for path_image in path_train:
  caption_list = captions_of_image[path_image]
  captions_list.extend(caption_list)
  image_list.extend([path_image] * len(caption_list))

encode_train = sorted(set(image_list))
cnn_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
embedded_input = cnn_model.input
hidden_layer = cnn_model.layers[-1].output
image_ds = tf.data.Dataset.from_tensor_slices(encode_train)
image_ds = image_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(16)
context_vector = tf.keras.Model(embedded_input, hidden_layer)

for image, path in image_ds:
  feature_map = context_vector(image)
  feature_map = tf.reshape(feature_map, (feature_map.shape[0], -1, feature_map.shape[3]))
  for fm, dir in zip(feature_map, path):
    feature_dir = dir.numpy().decode("utf-8")
    np.save(feature_dir, fm.numpy())

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000, oov_token="<unk>")
tokenizer.fit_on_texts(captions_list)

tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

tokenized_vector = tokenizer.texts_to_sequences(captions_list)
preprocessed_captions = tf.keras.preprocessing.sequence.pad_sequences(tokenized_vector, padding='post')
max_length = caption_length_max(tokenized_vector)

preprocessed_captions_list = collections.defaultdict(list)
for image, pre_cap in zip(image_list, preprocessed_captions):
  preprocessed_captions_list[image].append(pre_cap)

# Training and validation data splitting
caption_keys = list(preprocessed_captions_list.keys())
random.shuffle(caption_keys)

idx_ratio = int(len(caption_keys)*0.8)
train_image_title_keys, title_value_keys = caption_keys[:idx_ratio], caption_keys[idx_ratio:]

train_image_title = []
cap_train = []
for imaget in train_image_title_keys:
  capt_len = len(preprocessed_captions_list[imaget])
  train_image_title.extend([imaget] * capt_len)
  cap_train.extend(preprocessed_captions_list[imaget])

title_value = []
cap_val = []
for imagev in title_value_keys:
  capv_len = len(preprocessed_captions_list[imagev])
  title_value.extend([imagev] * capv_len)
  cap_val.extend(preprocessed_captions_list[imagev])

dataset = tf.data.Dataset.from_tensor_slices((train_image_title, cap_train))

dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.AUTOTUNE)

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

#Building attention class
class Attention(tf.keras.Model):
  def __init__(self, hidden_size):
    super(Attention, self).__init__()
    self.W1 = tf.keras.layers.Dense(hidden_size)
    self.W2 = tf.keras.layers.Dense(hidden_size)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    time_step = tf.expand_dims(hidden, 1)
    hidden_attention_layer = (tf.nn.tanh(self.W1(features) + self.W2(time_step)))
    score = self.V(hidden_attention_layer)
    weight_out = tf.nn.softmax(score, axis=1)
    context_vector = weight_out * features
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, weight_out

#Encoder class
class Encoder(tf.keras.Model):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embed_size)
        self.fc = tf.keras.layers.Dense(embed_size)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

#Decoder class
class Decoder(tf.keras.Model):
  def __init__(self, embed_size, hidden_size, vocabulary_size):
    super(Decoder, self).__init__()
    self.hidden_size = hidden_size
    self.embedding = tf.keras.layers.Embedding(vocabulary_size, embed_size)
    self.gru = tf.keras.layers.GRU(self.hidden_size, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.hidden_size)
    self.fc2 = tf.keras.layers.Dense(vocabulary_size)
    self.attention = Attention(self.hidden_size)

  def call(self, embed, features, hidden):
    context_vector, weight_out = self.attention(features, hidden)
    embed = self.embedding(embed)
    embed = tf.concat([tf.expand_dims(context_vector, 1), embed], axis=-1)
    output, state = self.gru(embed)
    embed = self.fc1(output)
    embed = tf.reshape(embed, (-1, x.shape[2]))
    embed = self.fc2(embed)
    return x, state, weight_out

  def init_zeros(self, batch_size):
    return tf.zeros((batch_size, self.hidden_size))

encoder = Encoder(embed_size)
decoder = Decoder(embed_size, hidden_size, vocabulary_size)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

start_epoch = 0
loss_plot = []

@tf.function
def train_model(image_tensor, target):
  loss = 0
  hidden = decoder.init_zeros(batch_size=target.shape[0])
  input_decoder = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)
  with tf.GradientTape() as tape:
      features = encoder(image_tensor)
      for i in range(1, target.shape[1]):
          predictions, hidden, _ = decoder(input_decoder, features, hidden)
          loss += calc_loss(target[:, i], predictions)
          input_decoder = tf.expand_dims(target[:, i], 1)
  total_loss = (loss / int(target.shape[1]))
  trainable_variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(gradients, trainable_variables))
  return loss, total_loss

steps = len(train_image_title)

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0
    for (batch, (image_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_model(image_tensor, target)
        total_loss += t_loss
    loss_plot.append(total_loss / steps)
    print('Epoch_num: {} , Loss_value: {}'.format(epoch+1, total_loss/steps))

plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss v/s Epochs')
plt.show()

def generate_results(image):
    plot_attn = np.zeros((max_length, attention_features_shape))
    hidden = decoder.init_zeros(batch_size=1)
    current_inp = tf.expand_dims(load_image(image)[0], 0)
    context_value = context_vector(current_inp)
    context_value = tf.reshape(context_value, (context_value.shape[0], -1, context_value.shape[3]))
    features = encoder(context_value)
    input_decoder = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []
    for i in range(max_length):
        predictions, hidden, weight_out = decoder(input_decoder, features, hidden)
        plot_attn[i] = tf.reshape(weight_out, (-1, )).numpy()
        index = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[index])
        if tokenizer.index_word[index] == '<end>':
            return result, plot_attn
        input_decoder = tf.expand_dims([index], 0)
    plot_attn = plot_attn[:len(result), :]
    return result, plot_attn

def generate_plots(image, caption_out, plot_attn):
    image_holder = np.array(Image.open(image))
    fig = plt.figure(figsize=(10, 10))
    length = len(caption_out)
    for i in range(length):
        resized_plot = np.resize(plot_attn[i], (8, 8))
        plot_size = max(np.ceil(length/2), 2)
        ax = fig.add_subplot(plot_size, plot_size, i+1)
        ax.set_title(caption_out[i])
        image = ax.imshow(image_holder)
        ax.imshow(resized_plot, cmap='gray', alpha=0.6, extent=image.get_extent())
    plt.tight_layout()
    plt.show()

random_num = np.random.randint(0, len(title_value))
image = title_value[random_num]
actual_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[random_num] if i not in [0]])
caption_out, plot_attn = generate_results(image)

print('Actual caption:', actual_caption)
print('Caption generated by attention model):', ' '.join(caption_out))
generate_plots(image, caption_out, plot_attn)

