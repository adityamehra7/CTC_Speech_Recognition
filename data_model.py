from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
import torchaudio
import torch
import torchvision


char_map_string = """
 ' 0
 '' 1
 a 2
 b 3
 c 4
 d 5
 e 6
 f 7
 g 8
 h 9
 i 10
 j 11
 k 12
 l 13
 m 14
 n 15
 o 16
 p 17
 q 18
 r 19
 s 20
 t 21
 u 22
 v 23
 w 24
 x 25
 y 26
 z 27
 """


class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = char_map_string
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map["''"]
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('', ' ')

train_audio_transforms = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35)
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

text_transform = TextTransform()


def preprocessing(batch,type='train'):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in batch:
        if type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths


class AsrDataModule(LightningDataModule):
    def __init__(self,batch_size,num_workers,pin_memory):
        super().__init__()
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        
    def setup(self,stage):
        self.train_set = torchaudio.datasets.LIBRISPEECH("./data",url="train-clean-100",download=True)
        self.test_set = torchaudio.datasets.LIBRISPEECH("./data",url="test-clean",download=True)
        self.val_set = torchaudio.datasets.LIBRISPEECH("./data",url="dev-clean",download=True)
        
    def train_dataloader(self):
        return DataLoader(dataset=self.train_set,batch_size=self.batch_size,
                          num_workers=self.num_workers,pin_memory=self.pin_memory,collate_fn= lambda x:preprocessing(x,type='train'))
        
    def val_dataloader(self):
        return DataLoader(self.val_set,batch_size=self.batch_size,
                          num_workers=self.num_workers,pin_memory=self.pin_memory,collate_fn= lambda x:preprocessing(x,type='train'))
        
    def test_dataloader(self):
        return DataLoader(self.val_set,batch_size=self.batch_size,
                          num_workers=self.num_workers,pin_memory=self.pin_memory,collate_fn=lambda x:preprocessing(x,type='train'))
        