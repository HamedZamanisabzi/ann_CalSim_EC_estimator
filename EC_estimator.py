# EC_estimator.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers.experimental.preprocessing import Normalization, IntegerLookup #CategoryEncoding
from tensorflow.keras.models import Model
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os

num_features = [
    'sac', 'exports', 'dcc', 'net_dcd', 'sjr', 'tide', 'smscg', 'sac_1d', 'exports_1d', 'dcc_1d', 
    'net_dcd_1d', 'sjr_1d', 'tide_1d', 'smscg_1d', 'sac_2d', 'exports_2d', 'dcc_2d', 'net_dcd_2d', 
    'sjr_2d', 'tide_2d', 'smscg_2d', 'sac_3d', 'exports_3d', 'dcc_3d', 'net_dcd_3d', 'sjr_3d', 'tide_3d', 
    'smscg_3d', 'sac_4d', 'exports_4d', 'dcc_4d', 'net_dcd_4d', 'sjr_4d', 'tide_4d', 'smscg_4d', 'sac_5d', 
    'exports_5d', 'dcc_5d', 'net_dcd_5d', 'sjr_5d', 'tide_5d', 'smscg_5d', 'sac_6d', 'exports_6d', 'dcc_6d', 
    'net_dcd_6d', 'sjr_6d', 'tide_6d', 'smscg_6d', 'sac_7d', 'exports_7d', 'dcc_7d', 'net_dcd_7d', 'sjr_7d', 
    'tide_7d', 'smscg_7d', 'sac_1ave', 'exports_1ave', 'dcc_1ave', 'net_dcd_1ave', 'sjr_1ave', 'tide_1ave', 
    'smscg_1ave', 'sac_2ave', 'exports_2ave', 'dcc_2ave', 'net_dcd_2ave', 'sjr_2ave', 'tide_2ave', 'smscg_2ave', 
    'sac_3ave', 'exports_3ave', 'dcc_3ave', 'net_dcd_3ave', 'sjr_3ave', 'tide_3ave', 'smscg_3ave', 'sac_4ave', 
    'exports_4ave', 'dcc_4ave', 'net_dcd_4ave', 'sjr_4ave', 'tide_4ave', 'smscg_4ave', 'sac_5ave', 'exports_5ave', 
    'dcc_5ave', 'net_dcd_5ave', 'sjr_5ave', 'tide_5ave', 'smscg_5ave', 'sac_6ave', 'exports_6ave', 'dcc_6ave', 
    'net_dcd_6ave', 'sjr_6ave', 'tide_6ave', 'smscg_6ave', 'sac_7ave', 'exports_7ave', 'dcc_7ave', 'net_dcd_7ave', 
    'sjr_7ave', 'tide_7ave', 'smscg_7ave', 'sac_8ave', 'exports_8ave', 'dcc_8ave', 'net_dcd_8ave', 'sjr_8ave', 
    'tide_8ave', 'smscg_8ave', 'sac_9ave', 'exports_9ave', 'dcc_9ave', 'net_dcd_9ave', 'sjr_9ave', 'tide_9ave', 
    'smscg_9ave', 'sac_10ave', 'exports_10ave', 'dcc_10ave', 'net_dcd_10ave', 'sjr_10ave', 'tide_10ave', 'smscg_10ave'
]

root_logdir = os.path.join(os.curdir, "tf_training_logs")

def root_mean_squared_error(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))

def load_data(file_name):
    return pd.read_csv(file_name)

def split_data(df, train_rows, test_rows):
    df_train = df.tail(train_rows)
    df_test = df.head(test_rows)
    return df_train, df_test

def build_model_inputs(df):
    inputs = []
    for feature in num_features:
        feature_input = Input(shape=(1,), name=f"{feature}_input")
        inputs.append(feature_input)
    return inputs

def preprocessing_layers(df, inputs):
    layers = []
    for feature in num_features:
        feature_layer = Normalization()
        feature_layer.adapt(df[feature].values.reshape(-1, 1))  
        layers.append(feature_layer(inputs[num_features.index(feature)]))
    return layers


def build_model(layers, inputs):  
    concatenated = tf.keras.layers.concatenate(layers)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=root_logdir)
    
    ann = tf.keras.models.Sequential()
    
    # First hidden layer with 8 neurons and sigmoid activation function
    ann.add(Dense(units=8, activation='sigmoid', input_dim=concatenated.shape[1], kernel_initializer="he_normal"))  
    ann.add(tf.keras.layers.BatchNormalization())
    
    # Second hidden layer with 2 neurons and sigmoid activation function
    ann.add(Dense(units=2, activation='sigmoid', kernel_initializer="he_normal")) 
    ann.add(tf.keras.layers.BatchNormalization())
    
    # Output layer with 1 neuron
    ann.add(Dense(units=1))  
    
    ann.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001), 
        loss=root_mean_squared_error, 
        metrics=['mean_absolute_error']
    )
    
    model = Model(inputs=inputs, outputs=ann(concatenated))
    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001), 
        loss=root_mean_squared_error, 
        metrics=['mean_absolute_error']
    )
    
    return model, tensorboard_cb


def train_model(model, tensorboard_cb, X_train, y_train, X_test, y_test):
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test), 
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=1000, 
            mode="min", 
            restore_best_weights=True), 
            tensorboard_cb
        ], 
        batch_size=128, 
        epochs=100, 
        verbose=0
    )
    return history, model
def calculate_metrics(model_name, y_train, y_train_pred, y_test, y_test_pred):
    y_train_np = y_train.values.ravel()
    y_train_pred_np = y_train_pred.ravel()

    # Calculate metrics for training data
    r2_train = r2_score(y_train_np, y_train_pred_np)
    rmse_train = np.sqrt(mean_squared_error(y_train_np, y_train_pred_np))
    percentage_bias_train = np.mean((y_train_pred_np - y_train_np) / y_train_np) * 100

    y_test_np = y_test.values.ravel()
    y_test_pred_np = y_test_pred.ravel()

    # Calculate metrics for test data
    r2_test = r2_score(y_test_np, y_test_pred_np)
    rmse_test = np.sqrt(mean_squared_error(y_test_np, y_test_pred_np))
    percentage_bias_test = np.mean((y_test_pred_np - y_test_np) / y_test_np) * 100

    # Return results as a dictionary
    return {
        'Model': model_name,
        'Train_R2': round(r2_train, 2),
        'Train_RMSE': round(rmse_train, 2),
        'Train_Percentage_Bias': round(percentage_bias_train, 2),
        'Test_R2': round(r2_test, 2),
        'Test_RMSE': round(rmse_test, 2),
        'Test_Percentage_Bias': round(percentage_bias_test, 2),
    }

def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()

def save_model(model, model_save_path):
    model.save(model_save_path)
    print(f"Model saved at location: {model_save_path}")

from tensorflow.keras.models import load_model

def load_model(model_path, loss_function):
    model = load_model(model_path, custom_objects={loss_function.__name__: loss_function})
    return model

def make_predictions(model, data, num_features):
    X_new = [data[feature] for feature in num_features]
    predictions = model.predict(X_new)
    return predictions






