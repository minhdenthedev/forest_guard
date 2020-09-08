import numpy as np

def extract_features(path): # Feature extracting
    import librosa
    import librosa.display
    
    # import matplotlib.pyplot as plt
    # Return audio time series and sampling rate
    audio, sample_rate = librosa.load(path, res_type = 'kaiser_fast')

    mfccs = librosa.feature.mfcc(y = audio, sr = sample_rate,  n_mfcc = 40) # return MFCC sequence

    mfcc_scaled = np.mean(mfccs.T, axis = 0)
    
    # return mfcc_scaled
    return mfcc_scaled
    # plt.figure(figsize = (3, 3))
    # # librosa.display.specshow(mfccs, x_axis = 'time')
    # librosa.display.specshow(mfccs)
    # # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig(f'./img/true/{counter:04d}.png')

    # plt.close()
    # plt.show()

counter = 0

if __name__ == '__main__':
    # path = '../data_splitted/true/0360.wav'
    # converted_path = './converted.wav'
    # stereo_to_mono(path, converted_path)
    # extract_features(converted_path)

    # print(extract_features(path).shape)

    import os
    path = 'data_splitted/false'
    file_names = os.listdir(path)

    x_false = np.empty((0,40))
    y_false = np.empty((0))

    for name in file_names:
        print(f'Processing {name} ({counter + 1}/{len(file_names)})')
        try:
            this_mfcc = extract_features(f'{path}/{name}')
        except Exception as e:
            print(f'Error processing {name}')
            print(e)
            continue

        x_false = np.append(x_false, np.asarray([this_mfcc]), axis = 0)
        y_false = np.append(y_false, 0)

        counter += 1

    print(x_false.shape)
    print(y_false.shape)
    print(y_false)
    np.save('x_false.npy', x_false)
    np.save('y_false.npy', y_false)