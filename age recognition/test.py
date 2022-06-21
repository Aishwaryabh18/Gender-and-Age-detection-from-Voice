# Data manipulation
import numpy as np
import matplotlib.pyplot as plt

# Feature extraction
import scipy
import librosa
import python_speech_features as mfcc
import os
from scipy.io.wavfile import read

# Model training
from sklearn.mixture import GaussianMixture as GMM
from sklearn import preprocessing
import pickle

# Live recording
import sounddevice as sd
import soundfile as sf

def get_MFCC(sr,audio):
    
    features = mfcc.mfcc(audio, sr, 0.025, 0.01, 13, appendEnergy = False)
    features = preprocessing.scale(features)
    
    return features

def get_features(source):
    
    files = [os.path.join(source,f) for f in os.listdir(source) if f.endswith('.wav')]
    
    features = []
    for f in files:
        sr,audio = read(f)
        vector   = get_MFCC(sr,audio)
        if len(features) == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

    return features

source_mdg = "./mdg"
features_mdg = get_features(source_mdg)

gmm_mdg = GMM(n_components = 8, max_iter=200, covariance_type='diag', n_init = 3)
gmm_mdg.fit(features_mdg)

source_yg = "./yg"
features_yg = get_features(source_yg)

gmm_yg = GMM(n_components = 8, max_iter=200, covariance_type='diag', n_init = 3)
gmm_yg.fit(features_yg)

source_ad = "./ad"
features_ad = get_features(source_ad)

gmm_ad = GMM(n_components = 8, max_iter=200, covariance_type='diag', n_init = 3)
gmm_ad.fit(features_ad)

# Save models
pickle.dump(gmm_mdg, open("mdg.gmm", "wb" ))
pickle.dump(gmm_yg, open("yg.gmm", "wb" ))
pickle.dump(gmm_ad, open("ad.gmm", "wb" ))

def record_and_predict(sr=16000, channels=1, duration=3, filename='7.wav'):
    
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=channels).reshape(-1)
    sd.wait()
    
    features = get_MFCC(sr,recording)
    scores = None

    log_likelihood_mdg = np.array(gmm_mdg.score(features)).sum()
    log_likelihood_yg = np.array(gmm_yg.score(features)).sum()
    log_likelihood_ad = np.array(gmm_ad.score(features)).sum()

    if (log_likelihood_mdg >= log_likelihood_yg and log_likelihood_mdg>= log_likelihood_ad):
        return("\n\n\n\n\n\n\n\nMiddle age")
    elif (log_likelihood_yg >= log_likelihood_ad and log_likelihood_yg>= log_likelihood_mdg):
        return("\n\n\n\n\n\n\n\nYoung")
    elif (log_likelihood_ad >= log_likelihood_mdg and log_likelihood_yg>= log_likelihood_yg):
        return("\n\n\n\n\n\n\n\nAdult")

find = record_and_predict()
print(find)