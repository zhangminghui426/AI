import os
import numpy as np
import librosa
import librosa.display
from scipy.io.wavfile import write

class WavProcess:
    """
    frame_t 25ms 帧长
    hop_length_t 10ms 步进
    win_length = int(frame_t*fs/1000)
    hop_length = int(hop_length_t*fs/1000)  帧移
    n_fft = int(2**np.ceil(np.log2(win_length)))  

    mute 默认去除静音部分 top_db = 30
    per_emphasis 默认进行预加重提升高频信息
    """

    def __init__(self, 
                 sr=16000, 
                 top_db=30,
                 win_length=512,
                 hop_length = 160,
                 n_fft = 512,
                 n_mels = 128,
                 n_mfcc = 20,
                 mute=True,
                 per_emphasis=True):
        super().__init__()
        self.sr = sr
        self.top_db = top_db
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.mute = mute
        self.per_emphasis = per_emphasis

    def load_wav(self, wav:str):
        file_type = wav.split('.')[-1]
        file_name = ''.join(wav.split('.')[:-1])
        if file_type != 'wav':
            old_wav = wav
            wav = file_name+'.wav'
            os.system("ffmpeg -i "+ old_wav + " -ac 1 -ar 16000"+" " + wav)
        y,fs = librosa.load(path=wav, sr=self.sr)
        if self.mute:
            y, index = librosa.effects.trim(y=y, top_db=self.top_db)
        return y, fs

    def split_wav(self, wav, window=2, split_path='./'):
        y, fs = self.load_wav(wav=wav)
        length = y.shape[0]
        cut = length // fs
        start = 0
        for i in range(0, cut, window):
            if fs * (i + window) > length:
                slipt_y = y[start:length]
            else:
                slipt_y = y[start:fs * (i + window)]
                start = fs * (i + window)
            
            write(os.path.join(split_path, '{}.wav'.format(i)), fs, slipt_y)

    def mfcc(self, wav:str):
        y,fs = self.load_wav(wav=wav)
        mfcc = librosa.feature.mfcc(y=y, 
                                    sr=fs, 
                                    n_mfcc=self.n_mfcc, 
                                    win_length = self.win_length,
                                    hop_length =self.hop_length,
                                    n_fft = self.n_fft,
                                    n_mels = self.n_mels,
                                    dct_type=1
                                    )
        # 一阶差分
        mfcc_deta =  librosa.feature.delta(data=mfcc)
        # 二阶差分
        mfcc_deta2 = librosa.feature.delta(data=mfcc, order=2)
        # 特征拼接
        mfcc_d1_d2 = np.concatenate([mfcc,mfcc_deta,mfcc_deta2],axis=0)
        return mfcc, mfcc_d1_d2
    
    def stft(self, wav:str):
        """
        shape D N
        D= n_fft/2+1
        N= len(y)/hop_length
        """
        
        y,fs = self.load_wav(wav=wav)
        if self.per_emphasis:
            y = librosa.effects.preemphasis(y)
        S = np.abs(librosa.stft(y=y, 
                                n_fft=self.n_fft, 
                                hop_length=self.hop_length, 
                                win_length=self.win_length))
        S = librosa.amplitude_to_db(S=S, ref=np.max)
        
        return S
    
    def fbank(self, wav:str):
        y,fs = self.load_wav(wav=wav)
        if self.mute:
            y, index = librosa.effects.trim(y=y, top_db=self.top_db)
        fb = librosa.feature.melspectrogram(y=y, 
                                            sr=fs, 
                                            n_fft=self.n_fft,
                                            win_length=self.win_length,
                                            hop_length=self.hop_length,
                                            n_mels=self.n_mels)
        fbank_db  = librosa.power_to_db(S=fb, ref=np.max)
        return fbank_db