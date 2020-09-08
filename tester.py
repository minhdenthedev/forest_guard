from tensorflow.keras.models import load_model
from tensorflow import newaxis

model = load_model("model_forest.h5")

import librosa
import numpy as np

audio, sample_rate = librosa.load('chainsaw.wav', res_type = 'kaiser_fast')
mfccs = librosa.feature.mfcc(y = audio, sr = sample_rate,  n_mfcc = 40)
mfcc_scaled = np.mean(mfccs.T, axis = 0)

x_test = mfcc_scaled[..., newaxis]

y_pred = model.predict(x = np.array([x_test]))
print(y_pred)