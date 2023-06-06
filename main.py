import pandas as pd
from sklearn.model_selection import train_test_split
import data_preprocessing as dp
import model_creation as mc
import model_training as mt
import numpy as np
from tensorflow.keras.models import load_model

def main():
    # Load the dataset
    df = pd.read_csv('/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv') # Change the path to the datafarme
    
    # Filter the dataset to keep the first 5000 images or however many you want to use for training and validation.
    df = df.head(5000) # keep the first 5000 images due to memory problems

    # Split the dataset into ships and no_ships dataframes
    ships_df = df[df['EncodedPixels'].notnull()]
    no_ships_df = df[df['EncodedPixels'].isnull()]
    
    # Split the ships and no_ships datasets into training and validation datasets
    train_ships_df, valid_ships_df = train_test_split(ships_df, test_size=0.2, random_state=42)
    train_no_ships_df, valid_no_ships_df = train_test_split(no_ships_df, test_size=0.2, random_state=42)

    # Concatenate ships and no_ships dataframes
    train_df = pd.concat([train_ships_df, train_no_ships_df])
    valid_df = pd.concat([valid_ships_df, valid_no_ships_df])

    # Load preprocessed data
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_valid = np.load('X_valid.npy')
    y_valid = np.load('y_valid.npy')
    
    # Load the model
    model = load_model('model_checkpoints.h5')

    # You could continue the training process using the loaded model if needed.
    # mt.train_model(model, X_train, y_train, X_valid, y_valid)

if __name__ == '__main__':
    main()
