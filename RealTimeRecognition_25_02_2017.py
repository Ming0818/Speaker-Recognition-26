# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 10:54:43 2017

@author: Arvind
"""

import sys
sys.path.append("C:/Users/Arvind/Anaconda3/pkgs/python_speech_features-0.4")

from python_speech_features import mfcc
import numpy as np
from scipy.cluster.vq import vq
import pyaudio
from gtts import gTTS
import os

vectDim = 12

c_asterix_MARV = np.loadtxt('F:/Speaker_Recognition/voiceModelParams/MARV.csv')
c_asterix_MNIT = np.loadtxt('F:/Speaker_Recognition/voiceModelParams/MNIT.csv')
c_asterix_MNAV = np.loadtxt('F:/Speaker_Recognition/voiceModelParams/MNAV.csv')
c_asterix_MAJA = np.loadtxt('F:/Speaker_Recognition/voiceModelParams/MAJA.csv')
c_asterix_MVEN = np.loadtxt('F:/Speaker_Recognition/voiceModelParams/MVEN.csv')
c_asterix_MBAL = np.loadtxt('F:/Speaker_Recognition/voiceModelParams/MBAL.csv')
c_asterix_MSRI = np.loadtxt('F:/Speaker_Recognition/voiceModelParams/MSRI.csv')
c_asterix_MNAN = np.loadtxt('F:/Speaker_Recognition/voiceModelParams/MNAN.csv')

def distCalc(mfccVect_t,c_t):
    
    codeIndex_t,distNearestCode_t = vq(np.transpose(mfccVect_t),np.transpose(c_t))
    dist = np.sum(np.power(distNearestCode_t,2))
    dist = dist/(vectDim*len(codeIndex_t))
    
    return(dist)


def featMatch(mfcc_t):
    
    dist_MARV = distCalc(mfcc_t,c_asterix_MARV)
    dist_MNIT = distCalc(mfcc_t,c_asterix_MNIT)
    dist_MNAV = distCalc(mfcc_t,c_asterix_MNAV)
    dist_MAJA = distCalc(mfcc_t,c_asterix_MAJA)
    dist_MVEN = distCalc(mfcc_t,c_asterix_MVEN)
    dist_MBAL = distCalc(mfcc_t,c_asterix_MBAL)
    dist_MSRI = distCalc(mfcc_t,c_asterix_MSRI)
    dist_MNAN = distCalc(mfcc_t,c_asterix_MNAN)
    
    lowestDist = min(dist_MARV,dist_MNIT,dist_MNAV,dist_MAJA,dist_MVEN,dist_MBAL,dist_MSRI,dist_MNAN)
    
    if lowestDist == dist_MARV:
         print("Arvind")
         tts = gTTS(text='Hi Arvind', lang='en')
         tts.save('test.mp3')
         os.system('start test.mp3')
         os.remove('test.mp3')
    elif lowestDist == dist_MNIT:
         print("Nithin")
         tts = gTTS(text='Hi Makaya', lang='en')
         tts.save('test.mp3')
         os.system('start test.mp3')
         os.remove('test.mp3')
    elif lowestDist == dist_MNAV:
         print("Naveed")
         tts = gTTS(text='Hi Naveed', lang='en')
         tts.save('test.mp3')
         os.system('start test.mp3')
         os.remove('test.mp3')
    elif lowestDist == dist_MAJA:
         print("Ajay")
         tts = gTTS(text='Hi Ajay', lang='en')
         tts.save('test.mp3')
         os.system('start test.mp3')
         os.remove('test.mp3')
         
    elif lowestDist == dist_MVEN:
         print("Venkat")
         tts = gTTS(text='Hi Venkat', lang='en')
         tts.save('test.mp3')
         os.system('start test.mp3')
         os.remove('test.mp3')
    elif lowestDist == dist_MBAL:
         print("Bala")
         tts = gTTS(text='Hi Bala', lang='en')
         tts.save('test.mp3')
         os.system('start test.mp3')
         os.remove('test.mp3')
    elif lowestDist == dist_MSRI:
         print("Sriharish")
         tts = gTTS(text='Hi Sriharsh', lang='en')
         tts.save('test.mp3')
         os.system('start test.mp3')
         os.remove('test.mp3')
    elif lowestDist == dist_MNAN:
         print("Nandy")
         tts = gTTS(text='Hi Nandy', lang='en')
         tts.save('test.mp3')
         os.system('start test.mp3')
         os.remove('test.mp3')
         
    print(dist_MARV)
    print(dist_MNIT)
    print(dist_MNAV)
    print(dist_MAJA)    
    print(dist_MVEN)
    print(dist_MBAL) 
    print(dist_MSRI)
    print(dist_MNAN)
    
    return
    
RATE=16000
RECORD_SECONDS = 5
CHUNKSIZE = 1024

input('Speak to recognise you')
     
# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

frames = [] # A python-list of chunks(numpy.ndarray)
for _ in range(0, int(RATE / CHUNKSIZE * RECORD_SECONDS)):
    data = stream.read(CHUNKSIZE)
    frames.append(np.fromstring(data, dtype=np.int16))

#Convert the list of numpy-arrays into a 1D array (column-wise)
testAudio = np.hstack(frames)

# close stream
stream.stop_stream()
stream.close()
p.terminate()

testAudio = np.array(testAudio)
mfcc_feat_testAudio = np.transpose(mfcc(testAudio,RATE))

featMatch(mfcc_feat_testAudio[1:13,:])
