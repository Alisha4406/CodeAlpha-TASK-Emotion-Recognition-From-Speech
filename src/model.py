# =========================================================
# MODEL BUILDING
# =========================================================

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization


def build_model(input_shape, num_classes):
    
    model = Sequential()
    
    model.add(Dense(256, activation="relu", input_shape=(input_shape,)))
    
    model.add(BatchNormalization())
    
    model.add(Dropout(0.3))
    
    
    model.add(Dense(128, activation="relu"))
    
    model.add(BatchNormalization())
    
    model.add(Dropout(0.3))
    
    
    model.add(Dense(64, activation="relu"))
    
    
    model.add(Dense(num_classes, activation="softmax"))
    
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    
    return model