import numpy as np
import time
import load_data
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, TimeDistributed
import matplotlib.pyplot as plt

noise_list = ['DKITCHEN', 'DWASHING', 'OOFFICE', 'NFIELD']  # Folder Names

batch_size = 16
number_batch = 2
lr = 5e-5
EPOCHS = 500

start = time.time()
data = load_data.Data(batch_size * number_batch, batch_size)

y_data = data.load_data()

noise_temp = data.make_noise(noise_list[0])
x_data = data.load_data(noise_temp)

y_data_temp = y_data
phase_temp = data.phase
for i in range(1, len(noise_list)):
    noise_temp = data.make_noise(noise_list[i])
    x_data_temp = data.load_data(noise_temp)
    x_data = np.concatenate((x_data, x_data_temp), axis=0)
    y_data = np.concatenate((y_data, y_data_temp), axis=0)
    data.phase = np.concatenate((data.phase, phase_temp), axis=0)

x_data /= data.regularization
x_data_temp = None
y_data_temp = None
phase_temp = None

x_data_test = x_data[:x_data.shape[0]//number_batch]
y_data_test = y_data[:y_data.shape[0]//number_batch]

x_data = x_data[x_data.shape[0]//number_batch:]
y_data = y_data[y_data.shape[0]//number_batch:]

print("Data Loading is Done! (", time.time() - start, ")")
print('Shape of train data(x,y):', x_data.shape, y_data.shape)
print('Shape of test data(x,y):', x_data_test.shape, y_data_test.shape)
print('Regularization:', data.regularization)
print(data.phase.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

class DenoiseGenerator(tf.keras.Model):
    def __init__(self):
        super(DenoiseGenerator, self).__init__()
        self.output_num = data.n_fft//2+1
        self.gru = tf.keras.layers.GRU(128, stateful=True, return_sequences=True)
        self.fc_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation='relu'))
        self.fc_2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.output_num, activation='relu'))

    def call(self, inputs):
        x = self.gru(inputs)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x

generator = DenoiseGenerator()

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

loss_object = tf.keras.losses.MeanSquaredError()

train_loss = tf.keras.metrics.Mean(name='train_loss')

train_losses = []

@tf.function
def train_step(noisy_wave, original_wave):
    with tf.GradientTape() as tape:
        denoise_wave = generator(noisy_wave, training=True)
        loss = loss_object(original_wave, denoise_wave)
    gradients = tape.gradient(loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

    train_loss(loss)

for epoch in range(EPOCHS):
    start = time.time()
    train_loss.reset_states()

    for x_wave, y_wave in train_dataset:
        train_step(x_wave, y_wave)

    train_losses.append(train_loss.result())

    print(f'Epoch {epoch + 1}, Train Loss: {train_loss.result()}, Time: {time.time() - start} sec')

    if (epoch != 0) and (epoch % (data.frame_num // data.truncate * number_batch) == 0):
        generator.reset_states()

generator.save_weights('generator_weights', save_format='tf')

# Plotting the learning curve
plt.plot(train_losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()