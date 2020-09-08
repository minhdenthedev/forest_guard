import pyaudio
import wave
import time
import numpy as np

import librosa

# initing constants
def init_recording():
    global CHUNK, SAMPLE_WIDTH, FORMAT, CHANNELS, SAMPLE_RATE
    CHUNK = 1024
    SAMPLE_WIDTH = 2
    FORMAT = 8
    CHANNELS = 2
    SAMPLE_RATE = 44100


def start_recording():
    audio = pyaudio.PyAudio()
    stream = audio.open(format = FORMAT,
                        channels = CHANNELS,
                        rate = SAMPLE_RATE,
                        input = True)

    stream.start_stream()

    frames = []

    time_start = time.time()

    while stream.is_active():
        data = stream.read(CHUNK)
        frames.append(data)
        # print('recording...')

        if (time.time() - time_start > 2):
            save_record(frames)
            # print('record saved')
            compress()

            # reset data stream
            frames = []
            time_start = time.time()

    stream.stop_stream()
    stream.close()

    audio.terminate()

def save_record(frames : list):
    wavefile = wave.open('record.wav', 'wb')
    wavefile.setnchannels(CHANNELS)
    wavefile.setframerate(SAMPLE_RATE)
    wavefile.setsampwidth(SAMPLE_WIDTH)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()

def compress():
    
    audio, sample_rate = librosa.load('record.wav', res_type = 'kaiser_fast')
    mfccs = librosa.feature.mfcc(y = audio, sr = sample_rate,  n_mfcc = 40)
    mfcc_scaled = np.mean(mfccs.T, axis = 0)
    # print(mfcc_scaled)

    print(analyze(mfcc_scaled))

def analyze(mfcc_scaled):
    x_test = mfcc_scaled[..., newaxis]

    y_pred = model.predict(x = np.array([x_test]))
    return y_pred

from tensorflow.keras.models import load_model
from tensorflow import newaxis

model = load_model("modelv1.h5")

if __name__ == '__main__':
    init_recording()
    start_recording()