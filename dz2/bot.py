import telebot
import onnxruntime as ort
import numpy as np
import pickle
import re
from scipy import signal
from scipy.io import wavfile
import numpy as np
from IPython.display import display, Audio
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as T
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import r2_score
from IPython.display import clear_output
from ipywidgets import interact
from PIL import Image
from sklearn.decomposition import PCA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return x*self.tanh(self.softplus(x))

class DenoisingAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(2, 256, kernel_size=3, stride=2, padding=1),
            Mish(),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            Mish(),
            nn.Conv1d(512, 1024, kernel_size=3, stride=2, padding=1),
            Mish(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, kernel_size=3, stride=2, padding=1),
            Mish(),
            nn.ConvTranspose1d(512, 256, kernel_size=3, stride=2, padding=1),
            Mish(),
            nn.ConvTranspose1d(256, 2, kernel_size=3, stride=2, padding=1),
        )


    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# sess = ort.InferenceSession("denoising_net.onnx")
# inputs = sess.get_inputs()
#ниже указать ранее сгенерированный bot_father'ом токен вашего бота
bot = telebot.TeleBot('6387696681:AAFwb3KJ_Ol3re6_rhikFKsly_pWS6HhP8I')

import ffmpeg
import os


@bot.message_handler(content_types=['voice'])
def get_message(message):
    file_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    
    with open('new_file.ogg', 'wb') as new_file:
        new_file.write(downloaded_file)
        
    os.system('C:/Users/hae19/ffmpeg.exe -y -i new_file.ogg -loglevel quiet -ac 1 -ar 16000 audio.wav')
    
    fs, data_noised = wavfile.read('audio.wav')
    data_noised = data_noised / (2**16-1)
    
    device = torch.device('cpu')
    model = torch.load('denoise.pt', map_location=device)
    model.eval()
    
    # noise = model(data)
    _, _, Zxx = signal.stft(data_noised, fs=fs, nperseg=512)
    _, xrec = signal.istft(Zxx, fs)
    
    X = np.concatenate([np.real(Zxx).T[:,:,None],
                    np.imag(Zxx).T[:,:,None]], axis=-1)
    normalization = X.reshape(-1, 2).std()
    X /= normalization
    tensor_x = torch.Tensor(np.transpose(X, [0, 2, 1])) # channels first
    pred = []
    batch_size = 128
    for i in tqdm(range(tensor_x.shape[0]//batch_size+1)):
        preds = model(tensor_x[i*batch_size:(i+1)*batch_size].to(device))
        pred.append(np.transpose(preds.detach().cpu().numpy(),
                                [0, 2, 1])*normalization)
    pred = np.concatenate(pred, axis=0)
    _, xrec = signal.istft((pred[:,:,0]+pred[:,:,1]*1j).T, fs)
    xrec = data_noised - xrec[:data_noised.size]
    
    # preds = model(tensor_x.to(device)).detach().cpu().numpy()
    # _, xrec = signal.istft((preds[:,:,0]+preds[:,:,1]*1j).T, fs)
    # xrec = data_noised - xrec[:data_noised.size]
    # print(preds)
    
    audio = Audio(xrec, rate=fs)
    
    with open('output.wav', 'wb') as f:
        f.write(audio.data)
    bot.send_audio(message.chat.id, audio=open('output.wav', 'rb'))
    
bot.polling(none_stop=True, interval=0)