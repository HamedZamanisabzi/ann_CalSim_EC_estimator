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


num_feature_dims = {"sac" : 18, "exports" : 18, "dcc": 18, "net_dcd" : 18, "sjr": 18, "tide" : 18, "smscg" : 18}

root_logdir = os.path.join(os.curdir, "tf_training_logs")


def feature_names():
    return list(num_feature_dims.keys())

def root_mean_squared_error(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))

def load_data(file_name):
    return pd.read_csv(file_name)

def split_data(df, train_rows, test_rows):
    df_train = df.tail(train_rows)
    df_test = df.head(test_rows)
    return df_train, df_test

def build_model_inputs_orig(df):
    inputs = []
    for feature in num_features:
        feature_input = Input(shape=(1,), name=f"{feature}_input")
        inputs.append(feature_input)
    return inputs

def build_model_inputs(df):
    inputs = []
    for feature,fdim in feature_names():
        feature_input = Input(shape=(fdim,), name=f"{feature}_input")
        inputs.append(feature_input)
    return inputs

def df_by_variable(df):
    indextups = []
    for col in list(df.columns):
        var = col
        lag = ""
        for key in num_feature_dims.keys():
            if col.startswith(key):
                var = key
                lag = col.replace(key,"").replace("_","")
                if lag is None or lag == "": lag = "0d"
                continue
        if var == "EC": lag = "0d"
        indextups.append((var,lag))
 
    ndx = pd.MultiIndex.from_tuples(indextups, names=('var', 'lag'))
    df.columns = ndx
    return df

def preprocessing_layers(df_var, inputs):
    layers = []
    for fndx,feature in enumerate(feature_names()):
        station_df = df_var.loc[:, pd.IndexSlice[feature,:]]
        feature_layer = Normalization()
        feature_layer.adapt(station_df.values.reshape(-1, num_feature_dims[feature]))  
        layers.append(feature_layer(inputs[fndx]))
    return layers


def preprocessing_layers1(df, inputs):
    layers = []
    for feature in num_features:
        feature_layer = Normalization()
        feature_layer.adapt(df[feature].values.reshape(-1, 1))  
        layers.append(feature_layer(inputs[num_features.index(feature)]))
    return layers


def build_model1(layers, inputs):  

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=root_logdir)
    x = tf.keras.layers.concatenate(layers)
    #ann = tf.keras.models.Sequential(name="EC")
    #ann.add(concatenated)
    
    # First hidden layer with 8 neurons and sigmoid activation function
    x = Dense(units=8, activation='sigmoid', input_dim=concatenated.shape[1], kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Second hidden layer with 2 neurons and sigmoid activation function
    x = Dense(units=2, activation='sigmoid', kernel_initializer="he_normal",name="hidden")(x) 
    x = tf.keras.layers.BatchNormalization(,name="batch_normalize")(x)
    
    # Output layer with 1 neuron
    output = Dense(units=1,name="emm_ec",activation="relu")(x)
    ann = Model(inputs = inputs, outputs = {"output" : outputs})

    ann.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001), 
        loss=root_mean_squared_error, 
        metrics=['mean_absolute_error']
    )
    
    #model = Model(inputs=inputs, outputs={"emm_ec": ann(concatenated)})
    #model = Model(inputs=inputs, outputs={"emm_ec": ann(concatenated)})
    #model.compile(
    #    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001), 
    #    loss=root_mean_squared_error, 
    #    metrics=['mean_absolute_error']
    #)
    
    print(ann.summary())
    return ann, tensorboard_cb


def build_model(layers, inputs):  

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=root_logdir)
    x = tf.keras.layers.concatenate(layers)
    
    # First hidden layer with 8 neurons and sigmoid activation function
    x = Dense(units=8, activation='sigmoid', input_dim=concatenated.shape[1], kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Second hidden layer with 2 neurons and sigmoid activation function
    x = Dense(units=2, activation='sigmoid', kernel_initializer="he_normal",name="hidden")(x) 
    x = tf.keras.layers.BatchNormalization(,name="batch_normalize")(x)
    
    # Output layer with 1 neuron
    output = Dense(units=1,name="emm_ec",activation="relu")(x)
    ann = Model(inputs = inputs, outputs = {"output" : output})

    ann.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001), 
        loss=root_mean_squared_error, 
        metrics=['mean_absolute_error']
    )
    
    return ann, tensorboard_cb



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






