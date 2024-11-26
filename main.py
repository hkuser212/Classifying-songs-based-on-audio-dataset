import os
import librosa
import librosa.display
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


y, sr = librosa.load('genres_original/reggae/reggae.00000.wav')
print ('y:',y)
print('y:shape',np.shape(y))
print('sample rate:',sr)
audio_file,_ = librosa.effects.trim(y)

print('Audio File:', audio_file, '\n')
print('Audio File shape:', np.shape(audio_file))
#plt.figure(figsize=(12, 4))
#librosa.display.waveshow(y=audio_file, sr=sr,color='r')
#plt.title('Reggae00000.wav')
#plt.show()
# fourier transform for signal
n_fft = 2048# fft window size
hop_length = 512# hop size
D = np.abs(librosa.stft(audio_file, n_fft=n_fft, hop_length=hop_length))
print('D:shape',np.shape(D))
plt.figure(figsize=(12, 4))
plt.plot(D)
plt.savefig('fourierofreggae.png')
# spectogram of audio file
DB = librosa.amplitude_to_db(D, ref=np.max)
plt.figure(figsize=(12, 4))
librosa.display.specshow(DB, sr=sr,hop_length=hop_length, x_axis='time', y_axis='hz')
plt.colorbar()
plt.savefig('spectogram of reggae.png')

y1,sr1 = librosa.load('genres_original/country/country.00037.wav')
y1,_ = librosa.effects.trim(y1)
print('y1:',y1)
print('y1 shape:',np.shape(y1))
S = librosa.feature.melspectrogram(y=y1, sr=sr1)
S_DB = librosa.power_to_db(S, ref=np.max)
plt.figure(figsize=(12, 4))
librosa.display.specshow(S_DB, sr=sr1, x_axis='time', y_axis='mel')
plt.colorbar()
plt.savefig('mel spectogram of country.png')



# Mel-Frequency Cepstral Coefficients:¶
# The Mel frequency cepstral coefficients (MFCCs) of a signal are a small set of features (usually about 10–20) which concisely describe the overall shape of a spectral envelope. It models the characteristics of the human voice.

mfccs = librosa.feature.mfcc(audio_file, sr=sr)
print('mfccs shape:', mfccs.shape)

#Displaying  the MFCCs:
plt.figure(figsize = (16, 6))
librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap = 'cool');
plt.colorbar()
plt.savefig('mfccs of reggae.png')