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
from sklearn import preprocessing


df = pd.read_csv("features_3_sec.csv")
print(df.head())
print(df.dtypes)
print(df.dropna(axis=0))
# boxplot
x = df[['label','tempo']]
f,ax = plt.subplots(figsize=(8,6))
sns.boxplot(x='label',y='tempo',data=df,palette="PRGn")
plt.title('bpm boxplot of genres')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('bpm boxplot of genres.png')
# correlation matrix for mean variable
corr = df.corr()
sns.heatmap(corr, cmap = 'RdBu_r', vmin = -1, vmax = 1, annot = True)
plt.title('Correlation matrix for mean variable')
plt.savefig('Correlation matrix for mean variable.png')
# PCA analysis on genres
data = df.iloc[0:,1:]
y = data['label']
X = data.drop('label',axis=1)

cols = X.columns
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled, columns = cols)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, y], axis = 1)

pca.explained_variance_ratio_
plt.figure(figsize = (16, 9))
sns.scatterplot(x = "principal component 1", y = "principal component 2", data = finalDf, hue = "label", alpha = 0.7,
               s = 100);

plt.title('PCA on Genres', fontsize = 25)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 10);
plt.xlabel("Principal Component 1", fontsize = 15)
plt.ylabel("Principal Component 2", fontsize = 15)
plt.savefig("PCA Scattert.jpg")

