import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

audio, sample_rate = librosa.load('chainsaw.wav', res_type = 'kaiser_fast')

mfccs = librosa.feature.mfcc(y = audio, sr = sample_rate,  n_mfcc = 40)
print('before transform ', mfccs[0][:10])
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(mfccs)
mfccs = scaler.transform(mfccs)
print(mfccs.shape)

# Reverse the experiment
data = scaler.inverse_transform(mfccs)

print('after inverse transform ', data[0][:10])

wav = librosa.feature.inverse.mfcc_to_audio(data)

'''
plt.title('MFCCs')
librosa.display.specshow(mfccs)
plt.tight_layout()
plt.show()
'''
