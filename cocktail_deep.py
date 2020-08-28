from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
import bci_workshop_tools as BCIw

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b,a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def preprocess_eeg(p):
   uf = butter_bandpass_filter(p,2,8,256)
   uf = BCIw.epoch(uf, 256,0.8 * 256)
   return uf

eeg_left = preprocess_eeg(np.load("mydata_left.npy"))
eeg_left1 = preprocess_eeg(np.load("mydata_left1.npy"))
eeg_left2 = preprocess_eeg(np.load("mydata_left2.npy"))
eeg_left3 = preprocess_eeg(np.load("mydata_left3.npy"))
eeg_left4 = preprocess_eeg(np.load("mydata_left4.npy"))

eeg_right = preprocess_eeg(np.load("mydata_right.npy"))
eeg_right1 = preprocess_eeg(np.load("mydata_right1.npy"))
eeg_right2 = preprocess_eeg(np.load("mydata_right2.npy"))
eeg_right3 = preprocess_eeg(np.load("mydata_right3.npy"))
eeg_right4 = preprocess_eeg(np.load("mydata_right4.npy"))

train_data = np.concatenate((eeg_left,eeg_left1,eeg_left2,eeg_left3,eeg_right,eeg_right1,eeg_right2,eeg_right3),axis=0)
test_data = np.concatenate((eeg_left4,eeg_right4),axis=0)

np.save("train_data",train_data)
np.save("test_data",test_data)











