import tensorflow as tf

# Check GPUs setup
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

