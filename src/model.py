from tensorflow import keras

def create_cnn_model(input_size=(128, 1), learning_rate=2e-4):
    input_layer = keras.layers.Input(shape=input_size, name="input_layer")

    # -------------------------------Block 1----------------------------
    x = keras.layers.Conv1D(64, (3), padding='same', kernel_initializer='he_normal', name='conv1.1')(input_layer)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv1D(64, (3), padding='same', kernel_initializer='he_normal', name='conv1.2')(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv1D(64, (3), padding='same', kernel_initializer='he_normal', name='conv1.3')(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPooling1D(4, strides=2, name='block1_pool')(x)
    x = keras.layers.Dropout(0.05)(x)

    # --------------------------Block 2----------------------------
    x = keras.layers.Conv1D(96, (3), padding='same', kernel_initializer='he_normal', name='conv2.1')(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv1D(96, (3), padding='same', kernel_initializer='he_normal', name='conv2.2')(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPooling1D(4, strides=2, name='block2_pool')(x)
    x = keras.layers.Dropout(0.1)(x)

    # --------------------------Block 3----------------------------
    x = keras.layers.Conv1D(128, (3), padding='same', kernel_initializer='he_normal', name='conv3.1')(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv1D(128, (3), padding='same', kernel_initializer='he_normal', name='conv3.2')(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPooling1D(4, strides=2, name='block3_pool')(x)
    x = keras.layers.Dropout(0.1)(x)

    # --------------------------Block 4----------------------------
    x = keras.layers.Conv1D(160, (3), padding='same', kernel_initializer='he_normal', name='conv4.1')(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv1D(160, (3), padding='same', kernel_initializer='he_normal', name='conv4.2')(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPooling1D(4, strides=2, name='block4_pool')(x)
    x = keras.layers.Dropout(0.1)(x)

    # --------------------------Block 5----------------------------
    x = keras.layers.Conv1D(192, (3), padding='same', kernel_initializer='he_normal', name='conv5.1')(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv1D(192, (3), padding='same', kernel_initializer='he_normal', name='conv5.2')(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPooling1D(4, strides=2, name='block5_pool')(x)
    x = keras.layers.Dropout(0.1)(x)

    # --------------------------Block 6----------------------------
    x = keras.layers.Conv1D(224, (3), padding='same', kernel_initializer='he_normal', kernel_regularizer='l2', name='conv6.1')(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2, strides=2, name='block6_pool')(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    output = keras.layers.Dense(1, activation="relu")(x)

    model = keras.Model(inputs=input_layer, outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)

    return model

