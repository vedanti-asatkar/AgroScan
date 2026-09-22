import tensorflow as tf

model = tf.keras.models.load_model('models/mobilenetv2_combined_v2.h5')
model.summary()