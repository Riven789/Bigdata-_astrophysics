import numpy as np
import h5py
import ml4gw.transforms.whitening
import torch
import os
import math
import tqdm

from ml4gw.transforms.whitening import FixedWhiten
from ml4gw.transforms import MultiResolutionSpectrogram, QScan
from torch.utils.data import Dataset
from embedding import Encoder, EncoderWrapper

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Spectrogram():
    def __init__(self, num_points, duration=8, sample_rate=2048, num_channels=2, n_fft=900, win_length=100, hop_length = 20):

        self.kernel_length = duration
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.whiten = FixedWhiten(num_channels=num_channels,kernel_length=self.kernel_length,sample_rate=sample_rate)
        

    def whitening(self, batch, fit_batch=None, fduration=1, fftlength=2, highpass=25, lowpass=200): # np.array([H_batch, L_batch])

        if fit_batch is None:
            fit_batch = batch
        #print(len(batch))
        self.whiten.fit(fduration, torch.from_numpy(fit_batch[0][0]), torch.from_numpy(fit_batch[1][0])
                        ,fftlength=fftlength,highpass=highpass,lowpass=lowpass)

        new_batch = self.whiten.forward(torch.from_numpy(batch).reshape(
            batch.shape[1],self.num_channels, batch.shape[-1]))

        self.kernel_length = int(batch.shape[-1]/self.sample_rate)

        #print(batch.shape, new_batch.shape)

        return new_batch


    def jitter(self, batch, params, shift_range=1):

        data_shifted = []
        theta_shifted = []
        data_unshifted = batch

        #print('batc', batch.shape)
    
        shifts = np.random.random(10)
        for shift in shifts:

            sample_rate = self.sample_rate
            front = int(shift*self.sample_rate)
            end = 2048*shift_range - front

            

            H = data_unshifted[0][math.floor(shift_range*shift*sample_rate):math.floor(-(1-shift)*shift_range*sample_rate)]
            L = data_unshifted[1][math.floor(shift_range*shift*sample_rate):math.floor(-(1-shift)*shift_range*sample_rate)]
            #print(H)
            data_shifted.append(np.vstack((H, L)))

        data_shifted = np.array(data_shifted)    
        theta_shifted = np.array(10*[params])

        #print(data_shifted.shape)

        return data_shifted, theta_shifted

    def spect(self, batch, n_fft=800, win=100, hop = 30):

        self.spectrogram = MultiResolutionSpectrogram(kernel_length = batch.shape[-1]//self.sample_rate,
                                                      sample_rate=self.sample_rate,
                                                      n_fft=[n_fft], win_length=[win], hop_length=[hop])

        spectrograms = self.spectrogram(torch.from_numpy(batch))

        cut = spectrograms.shape[-1]//2

        return spectrograms[:,:,10:90,cut:]
        

    def forward(self, batch, params):

        whitened_data = self.whitening(batch)

        #plt.plot(whitened_data[0][0])

        #print(whitened_data.shape)

        specs = []
        Theta = []

        for i in range(whitened_data.shape[0]):

            white = whitened_data[i]
            pars = params[i]
            #print('loop', white.shape)

            jittered, theta = self.jitter(white, pars)

            
            spec = self.spect(jittered)
            specs.append(spec)
            Theta.append(theta)
        

        return torch.from_numpy(np.array(specs)), torch.from_numpy(np.array(Theta))

data_dir = '/Users/shreyaggarwal/Desktop/Project8581/data/train/'

class SpectrogramParamDataset(Dataset):
    def __init__(self, data_dir=data_dir, num_files=None, weights=None, progress=False):

        self.data_dir=data_dir
        self.num_files=num_files

        self.load_data(progress)

        if weights is not None:
            
            self.embedding(weights)

    def __len__(self):
        return self.dataset.shape[0]


    def load_data(self, progress):
        files = [h5py.File(self.data_dir+file, 'r') 
                 for file in os.listdir(self.data_dir) if file[:11] == 'background-']

        if self.num_files is not None:
            files = files[:self.num_files]

        param_keys = ['chirp_mass', 'mass_ratio', 'distance', 'psi', 'phi', 'dec', 'snr']

        dataset = []
        theta = []

        load = range(len(files))
        if progress:
            load = tqdm.tqdm(load)
 
            
        for f, z in zip(files, load):

            batch = np.array([f['H1'], f['L1']])
            params = np.array([np.array(f[key]) for key in param_keys[:-1]])

            spec = Spectrogram(num_points=batch.shape[-1])

            data, params = spec.forward(batch, params.T)

            dataset.append(data)
            theta.append(params)

        self.dataset = torch.cat(dataset)
        self.theta = torch.cat(theta)

    def embedding(self, weights, embedding_dim=128):

        encoder1 = Encoder(embedding_dim)
        encoder2 = Encoder(embedding_dim)

        # Load the checkpoint
        checkpoint = torch.load(weights, map_location='cuda' 
                                if torch.cuda.is_available() else 'cpu')

        # Load weights
        encoder1.load_state_dict(checkpoint['encoder1_state_dict'])
        encoder2.load_state_dict(checkpoint['encoder2_state_dict'])

        encoder1.eval()
        encoder2.eval()


        wrapper = EncoderWrapper(encoder1, encoder2)
        print('em')
        with torch.no_grad():
            emb1, emb2 = wrapper(self.dataset)

        print('em')

        emb1 = emb1[:, 0] ## only one time shift
        emb2 = emb2[:, 0] 
        
        self.dataset =  torch.cat([emb1, emb2], dim=1)

        
        
        

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # return theta and data
        return (
            self.theta[idx].to(device=device),
            self.dataset[idx].to(device=device))



