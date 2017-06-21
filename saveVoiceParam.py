# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:44:10 2017

@author: Arvind
"""

import sys
sys.path.append("C:/Users/Arvind/Anaconda3/pkgs/python_speech_features-0.4")

from python_speech_features import mfcc
import scipy.io.wavfile as wavfile
import numpy as np
from scipy.cluster.vq import vq

vectDim = 12

def vectQuant(sourceVect,codeBookSize):
    
    # STEP 1
    e = 0.0001
    N = 1
    sourceVectSize = sourceVect.shape[1]
    c_asterix = np.zeros((vectDim,codeBookSize))
    c = np.zeros((vectDim,codeBookSize))  
    qOfx_m = np.zeros((vectDim,sourceVectSize))
    
    #STEP 2
    c_asterix[:,0] = np.transpose(np.mean(sourceVect,axis=1))
    D_asterix = 0
    D_curr = 0
    D_prev = D_asterix
    
    for i in range(sourceVectSize):
        D_asterix += np.sum(np.power(sourceVect[:,i]-c_asterix[:,0],2))  
    D_asterix = D_asterix/(sourceVectSize*vectDim)
    print('Initial Distortion --- %d' % (D_asterix))
    
    while (N<codeBookSize): # STEP 5 - Repeat steps 3 and 4 till desired number of codevectors are obtained
        
        print('outer loop')
        # STEP 3 - SPLITTING
        for i in range(N):          
            c[:,i] = (1+e)*c_asterix[:,i]
            c[:,N+i] = (1-e)*c_asterix[:,i]
            
        
        N = 2*N
        
        # STEP 4 - ITERATION
        D_prev = D_asterix 
        j = 0   # j - Iteration index
        
        # codeIndex - Codebook index for each source vector
        # distNearestCode - Distortion between source vector and its nearest code
        
        while True:
            codeIndex, distNearestCode = vq(np.transpose(sourceVect),np.transpose(c[:,0:N])) # Iteration - step (i)
            
            for i in range(sourceVectSize):
                qOfx_m[:,i] = c[:,codeIndex[i]]
            
            for i in range(N): # Iteration - Step (ii)
                c[:,i] = np.mean(sourceVect[:,np.nonzero(codeIndex==i)[0]],axis = 1)
            
           # print(c)
            
            j = j + 1       # Iteration - Step (iii)
            
            for i in range(sourceVectSize): # Iteration - Step (iv)
                D_curr += np.sum(np.power(sourceVect[:,i]-qOfx_m[:,i],2))
            D_curr = D_curr/(sourceVectSize*vectDim)
                
            eDash = (D_prev - D_curr)/D_prev
            D_prev = D_curr
            print('CodeBook Size --- %d ; Distortion of iteration %d --- %f' % (N,j,D_curr))
            
            if (eDash<e)and(j!=1):     # Iteration - Step (v)
                break
        
        D_asterix = D_curr  # Iteration - Step (vi)
        c_asterix = c
        
        #print(g)
             
    return(c_asterix)

    
a = ['a','b','c','d','e','f','g']    
    
# READ THE TRAINING SET
temp_1 = []
temp_2 = []
temp_3 = []
temp_4 = []
temp_5 = []
temp_6 = []
temp_7 = []
temp_8 = []
temp_9 = []
 
sigFAML = []
sigFDHH = []
sigMASM = []
sigMCBR = []
sigMARV = []
sigMNIT = []
sigMNAV = []
sigMAJA = []
sigMVEN = []

for i in a:
    (rate,temp_1) = wavfile.read("F:/Speaker_Recognition/elsdsr/train/FAML_S"+i+".wav")
    (rate,temp_2) = wavfile.read("F:/Speaker_Recognition/elsdsr/train/FDHH_S"+i+".wav")
    (rate,temp_3) = wavfile.read("F:/Speaker_Recognition/elsdsr/train/MASM_S"+i+".wav")
    (rate,temp_4) = wavfile.read("F:/Speaker_Recognition/elsdsr/train/MCBR_S"+i+".wav")
    #(rate,temp_5) = wavfile.read("F:/Speaker_Recognition/elsdsr/train_ourVoice/MARV_S"+i+".wav")
    (rate,temp_5) = wavfile.read("F:/Speaker_Recognition/elsdsr/noiseSolution/Arvind_"+i+".wav")
    (rate,temp_6) = wavfile.read("F:/Speaker_Recognition/elsdsr/noiseSolution/Nithin_"+i+".wav")
    (rate,temp_7) = wavfile.read("F:/Speaker_Recognition/elsdsr/noiseSolution/Naveed_"+i+".wav")
    (rate,temp_8) = wavfile.read("F:/Speaker_Recognition/elsdsr/noiseSolution/Aj_"+i+".wav")
    (rate,temp_9) = wavfile.read("F:/Speaker_Recognition/elsdsr/noiseSolution/Venkat_"+i+".wav")

    for j in range(len(temp_1)):
        sigFAML.append(temp_1[j])
    for j in range(len(temp_2)):
        sigFDHH.append(temp_2[j])
    for j in range(len(temp_3)):
        sigMASM.append(temp_3[j])
    for j in range(len(temp_4)):
        sigMCBR.append(temp_4[j])
    for j in range(len(temp_5)):
        sigMARV.append(temp_5[j])
    for j in range(len(temp_6)):
        sigMNIT.append(temp_6[j])
    for j in range(len(temp_7)):
        sigMNAV.append(temp_7[j])
    for j in range(len(temp_8)):
        sigMAJA.append(temp_8[j])
    for j in range(len(temp_9)):
        sigMVEN.append(temp_9[j])

sigFAML = np.array(sigFAML)
sigFDHH = np.array(sigFDHH)
sigMASM = np.array(sigMASM)
sigMCBR = np.array(sigMCBR)
sigMARV = np.array(sigMARV)
sigMNIT = np.array(sigMNIT)
sigMNAV = np.array(sigMNAV)
sigMAJA = np.array(sigMAJA)
sigMVEN = np.array(sigMVEN)

# FEATURE EXTRACTION
mfcc_feat_FAML = np.transpose(mfcc(sigFAML,rate))
mfcc_feat_FDHH = np.transpose(mfcc(sigFDHH,rate))
mfcc_feat_MASM = np.transpose(mfcc(sigMASM,rate))
mfcc_feat_MCBR = np.transpose(mfcc(sigMCBR,rate))
mfcc_feat_MARV = np.transpose(mfcc(sigMARV,rate))
mfcc_feat_MNIT = np.transpose(mfcc(sigMNIT,rate))
mfcc_feat_MNAV = np.transpose(mfcc(sigMNAV,rate))
mfcc_feat_MAJA = np.transpose(mfcc(sigMAJA,rate))
mfcc_feat_MVEN = np.transpose(mfcc(sigMVEN,rate))

# TRAINING SPEAKER MODEL
c_asterix_FAML = vectQuant(mfcc_feat_FAML[1:13,:],8)
c_asterix_FDHH = vectQuant(mfcc_feat_FDHH[1:13,:],8)
c_asterix_MASM = vectQuant(mfcc_feat_MASM[1:13,:],8)
c_asterix_MCBR = vectQuant(mfcc_feat_MCBR[1:13,:],8)
c_asterix_MARV = vectQuant(mfcc_feat_MARV[1:13,:],8)
c_asterix_MNIT = vectQuant(mfcc_feat_MNIT[1:13,:],8)
c_asterix_MNAV = vectQuant(mfcc_feat_MNAV[1:13,:],8)
c_asterix_MAJA = vectQuant(mfcc_feat_MAJA[1:13,:],8)
c_asterix_MVEN = vectQuant(mfcc_feat_MVEN[1:13,:],8)

np.savetxt('F:/Speaker_Recognition/voiceModelParams/FAML.csv',c_asterix_FAML)
np.savetxt('F:/Speaker_Recognition/voiceModelParams/FDHH.csv',c_asterix_FDHH)
np.savetxt('F:/Speaker_Recognition/voiceModelParams/MASM.csv',c_asterix_MASM)
np.savetxt('F:/Speaker_Recognition/voiceModelParams/MCBR.csv',c_asterix_MCBR)
np.savetxt('F:/Speaker_Recognition/voiceModelParams/MARV.csv',c_asterix_MARV)
np.savetxt('F:/Speaker_Recognition/voiceModelParams/MNIT.csv',c_asterix_MNIT)
np.savetxt('F:/Speaker_Recognition/voiceModelParams/MNAV.csv',c_asterix_MNAV)
np.savetxt('F:/Speaker_Recognition/voiceModelParams/MAJA.csv',c_asterix_MAJA)
np.savetxt('F:/Speaker_Recognition/voiceModelParams/MVEN.csv',c_asterix_MVEN)