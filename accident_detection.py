import cv2
import random
import pandas as pd
# Create sample data for the Bus Data with updated specifications
bus_data = {
    "Bus ID": [f"B{i+1}" for i in range(100)],  # 100 unique Bus IDs
    "Depot ID": [f"D{random.randint(1, 5)}" for _ in range(100)],  # Random depot assignment (D1 to D5)
    "Route ID": [f"R{random.randint(1, 8)}" for _ in range(100)],  # Random route assignment (R1 to R8)
    "Bus Type": [random.choice(["Deluxe", "Normal"]) for _ in range(100)],  # Random assignment of Bus Type
    "License Plate Number": [f"AB{random.randint(1000, 9999)}" for _ in range(100)],  # Random license plates
    "Status": [random.choice(["In Service", "Under Maintenance", "Reserved"]) for _ in range(100)]  # Random status
}

# Convert to DataFrame
bus_df = pd.DataFrame(bus_data)

# Save to CSV
bus_df.to_csv('/content/drive/My Drive/df.csv',index=False)
import warnings
warnings.filterwarnings("ignore")


import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import kagglehub

# Download latest version
path = kagglehub.dataset_download("ckay16/accident-detection-from-cctv-footage")

tr_data_dir = os.path.join("/content/drive/My Drive/dataset/train")
tr_data = tf.keras.utils.image_dataset_from_directory(
                            tr_data_dir,image_size=(256, 256),
                            seed = 12332
                            ) 
tr_data_iterator = tr_data.as_numpy_iterator()
tr_batch = tr_data_iterator.next()
def label_to_category(label):
    if(label == 1):
        return "No Accident"
    elif label == 0:
        return "Accident"
    else :
        return "error"
tr_data = tr_data.map(lambda x,y: (x/255, y))

print("Max pixel value : ",tr_batch[0].max())
print("Min pixel value : ",tr_batch[0].min())
val_data_dir = os.path.join("/content/drive/MyDrive/dataset/test")
val_data = tf.keras.utils.image_dataset_from_directory(val_data_dir)
val_data_iterator = val_data.as_numpy_iterator()
val_batch = val_data_iterator.next()
val_data = val_data.map(lambda x,y: (x/255, y))
val_batch = val_data.as_numpy_iterator().next()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
# Adding neural Layer
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(tr_data, epochs=20, validation_data=val_data, callbacks=[tensorboard_callback])
model.save("//content/drive/MyDrive/accidents.keras")                                                     
import cv2

# load random samples from samples directory
#random_data_dirname = os.path.join("/content/drive/MyDrive/dataset/test/Accident")
#pics = [os.path.join(random_data_dirname, filename) for filename in os.listdir(random_data_dirname)]

# load first file from samples
sample = cv2.imread("/content/drive/MyDrive/dataset/sam/accident.jpeg", cv2.IMREAD_COLOR)
sample = cv2.resize(sample, (256, 256))

prediction = 1 - model.predict(np.expand_dims(sample/255, 0))

if prediction >= 0.5: 
    label = 'Predicted class is Accident'
else:
    label = 'Predicted class is Not Accident'

plt.title(label)
plt.imshow(sample)
plt.show()
