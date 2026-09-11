import tensorflow as tf
from tensorflow.keras import layers

IMG=128; BATCH=32

train_ds = tf.keras.utils.image_dataset_from_directory("dataset/augmented", image_size=(IMG,IMG), batch_size=BATCH, shuffle=True)
val_ds = tf.keras.utils.image_dataset_from_directory("dataset/processed/val", image_size=(IMG,IMG), batch_size=BATCH)

print("Classes:", train_ds.class_names) # should be bike, bus, car, truck

train_ds = train_ds.map(lambda x,y: (x/255., y))
val_ds = val_ds.map(lambda x,y: (x/255., y))

model = tf.keras.Sequential([
    layers.Conv2D(32,3,activation='relu',padding='same',input_shape=(IMG,IMG,3)),
    layers.Conv2D(32,3,activation='relu',padding='same'), layers.MaxPooling2D(),
    layers.Conv2D(64,3,activation='relu',padding='same'),
    layers.Conv2D(64,3,activation='relu',padding='same'), layers.MaxPooling2D(),
    layers.Conv2D(128,3,activation='relu',padding='same'),
    layers.Conv2D(128,3,activation='relu',padding='same'), layers.MaxPooling2D(),
    layers.Flatten(), layers.Dense(128,activation='relu'), layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(train_ds, validation_data=val_ds, epochs=10)
model.save("model_vgg7_final.h5")
print("MODEL SAVED - PROJECT FINISHED!")