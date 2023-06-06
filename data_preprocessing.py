def rle_decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


from tensorflow.keras.preprocessing.image import img_to_array

from keras.utils import to_categorical

border = 5
im_chan = 3
n_classes = 2 

def preprocess_data(img_ids, img_dir, df, train=True):
    """This function preprocesses the image and mask data"""
    X = np.zeros((len(img_ids), 256, 256, im_chan), dtype=np.uint8)  # changed dimensions here
    y = np.zeros((len(img_ids), 256, 256, n_classes), dtype=np.uint8)  # changed dimensions here
    for n, id_ in enumerate(img_ids):
        
        img_path = img_dir + id_
        img = cv2.imread(img_path)
        
#         if img is not None:
#             print(f"Image {id_} loaded successfully.")
#         else:
#             print(f"Failed to load image {id_}.")
        img = cv2.resize(img, (256, 256))  
#         print(f"Resized image shape: {img.shape}")  # Print the dimensions of the resized image
        X[n] = img

        if train:

            mask = np.zeros((768, 768))
            masks = df.loc[df['ImageId'] == id_, 'EncodedPixels'].tolist()


            if masks[0] != masks[0]:

                pass
            else:
                for mask_ in masks:
                    mask += rle_decode(mask_)
            

            mask = cv2.resize(mask, (256, 256))  # added resizing here
#             print(f"Resized mask shape: {mask.shape}") 

            mask = np.expand_dims(mask, axis=-1)
#             print(f"Expanded mask shape: {mask.shape}") 
            mask_cat = to_categorical(mask, num_classes=n_classes)
    
#             print(f"Categorical mask shape: {mask_cat.shape}")
            y[n, ...] = mask_cat.squeeze()

    return X, y


img_dir = '/kaggle/input/airbus-ship-detection/train_v2/'


train_ids = train_df['ImageId'].values
valid_ids = valid_df['ImageId'].values


import os

if os.path.exists('X_train.npy') and os.path.exists('y_train.npy'):

    print("Loading data...")
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
else:

    print("Preprocessing data...")
    X_train, y_train = preprocess_data(train_ids, img_dir, train_df)


    print("Saving data...")
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)


if os.path.exists('X_valid.npy') and os.path.exists('y_valid.npy'):
    print("Loading data...")
    X_valid = np.load('X_valid.npy')
    y_valid = np.load('y_valid.npy')
else:
    print("Preprocessing data...")
    X_valid, y_valid = preprocess_data(valid_ids, img_dir, valid_df)

    print("Saving data...")
    np.save('X_valid.npy', X_valid)
    np.save('y_valid.npy', y_valid)

