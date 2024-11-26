













import os
import librosa
import librosa.display
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# Load the audio file
y, sr = librosa.load('genres_original/reggae/reggae.00000.wav')
print('Audio signal shape:', np.shape(y))
print('Sample rate:', sr)

# Trim silent edges
audio_file, _ = librosa.effects.trim(y)
print('Trimmed audio signal shape:', np.shape(audio_file))

# Plot waveform
plt.figure(figsize=(12, 4))
librosa.display.waveshow(audio_file, sr=sr, color='b')
plt.title('Waveform of reggae.00000.wav')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig('waveform_reggae.png')
plt.show()

# Fourier Transform and Spectrogram
n_fft = 2048  # FFT window size
hop_length = 512  # Hop size
D = np.abs(librosa.stft(audio_file, n_fft=n_fft, hop_length=hop_length))
print('STFT shape:', np.shape(D))

# Plot spectrogram
DB = librosa.amplitude_to_db(D, ref=np.max)
plt.figure(figsize=(12, 4))
librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
plt.colorbar()
plt.title('Spectrogram of reggae.00000.wav')
plt.savefig('spectrogram_reggae.png')
plt.show()

# Extracting Mel Spectrogram
S = librosa.feature.melspectrogram(y=audio_file, sr=sr, n_fft=n_fft, hop_length=hop_length)
S_DB = librosa.power_to_db(S, ref=np.max)
plt.figure(figsize=(12, 4))
librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram of reggae.00000.wav')
plt.savefig('mel_spectrogram_reggae.png')
plt.show()

# MFCCs (Mel Frequency Cepstral Coefficients)
mfccs = librosa.feature.mfcc(y=audio_file, sr=sr, n_mfcc=13)
print('MFCCs shape:', mfccs.shape)
plt.figure(figsize=(12, 4))
librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap='cool')
plt.colorbar()
plt.title('MFCCs of reggae.00000.wav')
plt.savefig('mfccs_reggae.png')
plt.show()

# Spectral Centroid
spectral_centroid = librosa.feature.spectral_centroid(y=audio_file, sr=sr)
plt.figure(figsize=(12, 4))
frames = range(len(spectral_centroid[0]))
t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
plt.semilogy(t, spectral_centroid[0], label='Spectral Centroid')
plt.xlabel('Time (s)')
plt.ylabel('Hz')
plt.title('Spectral Centroid of reggae.00000.wav')
plt.legend()
plt.savefig('spectral_centroid_reggae.png')
plt.show()

# Chroma Features
chroma = librosa.feature.chroma_stft(y=audio_file, sr=sr, hop_length=hop_length)
plt.figure(figsize=(12, 4))
librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', cmap='coolwarm')
plt.colorbar()
plt.title('Chroma Features of reggae.00000.wav')
plt.savefig('chroma_features_reggae.png')
plt.show()

# Spectral Contrast
spectral_contrast = librosa.feature.spectral_contrast(y=audio_file, sr=sr)
plt.figure(figsize=(12, 4))
librosa.display.specshow(spectral_contrast, sr=sr, x_axis='time', y_axis='hz', cmap='coolwarm')
plt.colorbar()
plt.title('Spectral Contrast of reggae.00000.wav')
plt.savefig('spectral_contrast_reggae.png')
plt.show()

# Zero-Crossing Rate
zero_crossings = librosa.feature.zero_crossing_rate(y=audio_file)
plt.figure(figsize=(12, 4))
plt.plot(t, zero_crossings[0], label='Zero-Crossing Rate')
plt.xlabel('Time (s)')
plt.ylabel('Rate')
plt.title('Zero-Crossing Rate of reggae.00000.wav')
plt.legend()
plt.savefig('zero_crossing_rate_reggae.png')
plt.show()

# Spectral Roll-off
rolloff = librosa.feature.spectral_rolloff(y=audio_file, sr=sr, roll_percent=0.85)
plt.figure(figsize=(12, 4))
plt.semilogy(t, rolloff[0], label='Spectral Roll-off')
plt.xlabel('Time (s)')
plt.ylabel('Hz')
plt.title('Spectral Roll-off of reggae.00000.wav')
plt.legend()
plt.savefig('spectral_rolloff_reggae.png')
plt.show()

# Root-Mean-Square Energy
rms = librosa.feature.rms(y=audio_file)
plt.figure(figsize=(12, 4))
plt.plot(t, rms[0], label='RMS Energy', color='r')
plt.xlabel('Time (s)')
plt.ylabel('Energy')
plt.title('RMS Energy of reggae.00000.wav')
plt.legend()
plt.savefig('rms_energy_reggae.png')
plt.show()























# import os
# import librosa
# import librosa.display
# import numpy as np
# # import warnings
# warnings.filterwarnings('ignore')
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
#
#
# y, sr = librosa.load('genres_original/reggae/reggae.00000.wav')
# print ('y:',y)
# print('y:shape',np.shape(y))
# print('sample rate:',sr)
# audio_file,_ = librosa.effects.trim(y)
#
# print('Audio File:', audio_file, '\n')
# print('Audio File shape:', np.shape(audio_file))
# #plt.figure(figsize=(12, 4))
# #librosa.display.waveshow(y=audio_file, sr=sr,color='r')
# #plt.title('Reggae00000.wav')
# #plt.show()
# # fourier transform for signal
# n_fft = 2048# fft window size
# hop_length = 512# hop size
# D = np.abs(librosa.stft(audio_file, n_fft=n_fft, hop_length=hop_length))
# print('D:shape',np.shape(D))
# plt.figure(figsize=(12, 4))
# plt.plot(D)
# plt.savefig('fourierofreggae.png')
# # spectogram of audio file
# DB = librosa.amplitude_to_db(D, ref=np.max)
# plt.figure(figsize=(12, 4))
# librosa.display.specshow(DB, sr=sr,hop_length=hop_length, x_axis='time', y_axis='hz')
# plt.colorbar()
# plt.savefig('spectogram of reggae.png')
#
# y1,sr1 = librosa.load('genres_original/country/country.00037.wav')
# y1,_ = librosa.effects.trim(y1)
# print('y1:',y1)
# print('y1 shape:',np.shape(y1))
# S = librosa.feature.melspectrogram(y=y1, sr=sr1)
# S_DB = librosa.power_to_db(S, ref=np.max)
# plt.figure(figsize=(12, 4))
# librosa.display.specshow(S_DB, sr=sr1, x_axis='time', y_axis='mel')
# plt.colorbar()
# plt.savefig('mel spectogram of country.png')
#
#
#
# # Mel-Frequency Cepstral Coefficients:¶
# # The Mel frequency cepstral coefficients (MFCCs) of a signal are a small set of features (usually about 10–20) which concisely describe the overall shape of a spectral envelope. It models the characteristics of the human voice.
#
# mfccs = librosa.feature.mfcc(audio_file, sr=sr)
# print('mfccs shape:', mfccs.shape)
#
# #Displaying  the MFCCs:
# plt.figure(figsize = (16, 6))
# librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap = 'cool');
# plt.colorbar()
# plt.savefig('mfccs of reggae.png')