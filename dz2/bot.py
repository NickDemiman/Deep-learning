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

sess = ort.InferenceSession("denoising_net.onnx")
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
        
    os.system('C:/Users/hae19/ffmpeg.exe -y -i new_file.ogg -ac 1 -ar 16000 audio.wav')
    
    fs, data = wavfile.read('audio.wav')
    data = data / (2**16-1)
    data[0] = 1
    print(data.shape)
    print(sess._inputs_meta)
    # sess.run(None, {'input': torch.rand(1, 2, )})[0]
    bot.send_audio(message.chat.id, audio=open('output.wav', 'rb'))
    
bot.polling(none_stop=True, interval=0)