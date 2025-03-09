import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)
import os
import numpy as np # Numpy <3
import soundfile as sf # WAV read + write
import scipy.interpolate as interp # Interpolator for feats
import resampy # Resampler (as in sampling rate stuff)
from pathlib import Path # path manipulation
import re
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
import onnxruntime
from wav2mel import PitchAdjustableMelSpectrogram
import dataclasses
import soundfile as sf
import numpy as np
import torch

version = '0.0.2-ccb'
help_string = '''usage: 🐱haicmimisampler🐱 in_file out_file pitch velocity [flags] [offset] [length] [consonant] [cutoff] [volume] [modulation] [tempo] [pitch_string]

Resamples using the PC-NSF-HIFIGAN Vocoder.

arguments:
\tin_file\t\tPath to input file.
\tout_file\tPath to output file.
\tpitch\t\tThe pitch to render on.
\tvelocity\tThe consonant velocity of the render.

optional arguments:
\tflags\t\tThe flags of the render.
\toffset\t\tThe offset from the start of the render area of the sample. (default: 0)
\tlength\t\tThe length of the stretched area in milliseconds. (default: 1000)
\tconsonant\tThe unstretched area of the render in milliseconds. (default: 0)
\tcutoff\t\tThe cutoff from the end or from the offset for the render area of the sample. (default: 0)
\tvolume\t\tThe volume of the render in percentage. (default: 100)
\tmodulation\tThe pitch modulation of the render in percentage. (default: 0)
\ttempo\t\tThe tempo of the render. Needs to have a ! at the start. (default: !100)
\tpitch_string\tThe UTAU pitchbend parameter written in Base64 with RLE encoding. (default: AA)'''

notes = {'C' : 0, 'C#' : 1, 'D' : 2, 'D#' : 3, 'E' : 4, 'F' : 5, 'F#' : 6, 'G' : 7, 'G#' : 8, 'A' : 9, 'A#' : 10, 'B' : 11} # Note names lol
note_re = re.compile(r'([A-G]#?)(-?\d+)') # Note Regex for conversion
cache_ext = '.hjm.npz' # cache file extension

# Flags
flags = ['fe', 'fl', 'fo', 'fv', 'fp', 've', 'vo', 'g', 't', 'A', 'B', 'G', 'P', 'S', 'p', 'R', 'D', 'C', 'Z']
flag_re = '|'.join(flags)
flag_re = f'({flag_re})([+-]?\\d+)?'
flag_re = re.compile(flag_re)

@dataclasses.dataclass
class Config:
    sample_rate: int = 44100  # UTAU only really likes 44.1khz
    win_size: int = 2048
    hop_size: int = 512
    extract_hop_size: int = 128
    n_mels: int = 128
    n_fft: int = 2048
    mel_fmin: float = 40
    mel_fmax: float = 16000
    f0_extractor: str = 'parselmouth'
    f0_min: float = 65
    f0_max: float = 1600
    vocoder_path: str = r"pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx"

ort_session = onnxruntime.InferenceSession(Config.vocoder_path,providers=['DmlExecutionProvider', 'CPUExecutionProvider'])

melAnalysis_extract_hop_size = PitchAdjustableMelSpectrogram(
    sample_rate=Config.sample_rate, 
    n_fft=Config.n_fft, 
    win_length=Config.win_size, 
    hop_length=Config.extract_hop_size, 
    f_min=Config.mel_fmin, 
    f_max=Config.mel_fmax,
    n_mels=Config.n_mels
    )

melAnalysis = PitchAdjustableMelSpectrogram(
    sample_rate=Config.sample_rate, 
    n_fft=Config.n_fft, 
    win_length=Config.win_size, 
    hop_length=Config.hop_size, 
    f_min=Config.mel_fmin, 
    f_max=Config.mel_fmax,
    n_mels=Config.n_mels
    )

def dynamic_range_compression_torch(x, C=1, clip_val=1e-9):
    return torch.log(torch.clamp(x, min=clip_val) * C)

#mel to wave
def m2w(f0,mel):
    '''
    mel shape = (n_mels, n_frames*scaling_ratio)
    f0 shape = (n_frames*scaling_ratio,)
    vocoder_keyshift shape = (n_frames,)
    '''

    # 加载vocoder模型
    print(onnxruntime.get_available_providers(),"00000000000") 

    # 准备输入
    #mel从tensor转numpy
    #f0从float64转32
    f0 = f0.astype(np.float32)
    mel = mel.astype(np.float32)
    print(f'mel dtype: {mel.dtype}')
    print(f'f0 dtype: {f0.dtype}')
    print(f'mel shape: {mel.shape}')
    print(f'f0 shape: {f0.shape}')
    #给mel和f0添加batched维度
    #print(mel)
    mel = np.expand_dims(mel, axis=0).transpose(0, 2, 1)
    f0 = np.expand_dims(f0, axis=0)
    input_data = {
        'mel': mel,
        'f0': f0,
    }

    # 执行模型
    output = ort_session.run(['waveform'], input_data)[0]

    # 输出
    wave = output[0]
    return wave






# Pitch string interpreter
def to_uint6(b64):
    """Convert one Base64 character to an unsigned integer.

    Parameters
    ----------
    b64 : str
        The Base64 character.

    Returns
    -------
    int
        The equivalent of the Base64 character as an integer.
    """
    c = ord(b64) # Convert based on ASCII mapping
    if c >= 97:
        return c - 71
    elif c >= 65:
        return c - 65
    elif c >= 48:
        return c + 4
    elif c == 43:
        return 62
    elif c == 47:
        return 63
    else:
        raise Exception

def to_int12(b64):
    """Converts two Base64 characters to a signed 12-bit integer.

    Parameters
    ----------
    b64 : str
        The Base64 string.

    Returns
    -------
    int
        The equivalent of the Base64 characters as a signed 12-bit integer (-2047 to 2048)
    """
    uint12 = to_uint6(b64[0]) << 6 | to_uint6(b64[1]) # Combined uint6 to uint12
    if uint12 >> 11 & 1 == 1: # Check most significant bit to simulate two's complement
        return uint12 - 4096
    else:
        return uint12

def to_int12_stream(b64):
    """Converts a Base64 string to a list of integers.

    Parameters
    ----------
    b64 : str
        The Base64 string.

    Returns
    -------
    list
        The equivalent of the Base64 string if split every 12-bits and interpreted as a signed 12-bit integer.
    """
    res = []
    for i in range(0, len(b64), 2):
        res.append(to_int12(b64[i:i+2]))
    return res

def pitch_string_to_cents(x):
    """Converts UTAU's pitchbend argument to an ndarray representing the pitch offset in cents.

    Parameters
    ----------
    x : str
        The pitchbend argument.

    Returns
    -------
    ndarray
        The pitchbend argument as pitch offset in cents.
    """
    pitch = x.split('#') # Split RLE Encoding
    res = []
    for i in range(0, len(pitch), 2):
        # Go through each pair
        p = pitch[i:i+2]
        if len(p) == 2:
            # Decode pitch string and extend RLE
            pitch_str, rle = p
            res.extend(to_int12_stream(pitch_str))
            res.extend([res[-1]] * int(rle))
        else:
            # Decode last pitch string without RLE if it exists
            res.extend(to_int12_stream(p[0]))
    res = np.array(res, dtype=np.int32)
    if np.all(res == res[0]):
        return np.zeros(res.shape)
    else:
        return np.concatenate([res, np.zeros(1)])

# Pitch conversion
def note_to_midi(x):
    """Note name to MIDI note number."""
    note, octave = note_re.match(x).group(1, 2)
    octave = int(octave) + 1
    return octave * 12 + notes[note]

def midi_to_hz(x):
    """MIDI note number to Hertz using equal temperament. A4 = 440 Hz."""
    return 440 * np.exp2((x - 69) / 12)

# WAV read/write
def read_wav(loc):
    """Read audio files supported by soundfile and resample to 44.1kHz if needed. Mixes down to mono if needed.

    Parameters
    ----------
    loc : str or file
        Input audio file.

    Returns
    -------
    ndarray
        Data read from WAV file remapped to [-1, 1] and in 44.1kHz
    """
    if type(loc) == str: # make sure input is Path
        loc = Path(loc)

    exists = loc.exists()
    if not exists: # check for alternative files
        for ext in sf.available_formats().keys():
            loc = loc.with_suffix('.' + ext.lower())
            exists = loc.exists()
            if exists:
                break

    if not exists:
        raise FileNotFoundError("No supported audio file was found.")
    
    x, fs = sf.read(str(loc))
    if len(x.shape) == 2:
        # Average all channels... Probably not too good for formats bigger than stereo
        x = np.mean(x, axis=1)

    if fs != Config.sample_rate:
        x = resampy.resample(x, fs, Config.sample_rate)

    return x

def save_wav(loc, x):
    """Save data into a WAV file.

    Parameters
    ----------
    loc : str or file
        Output WAV file.

    x : ndarray
        Audio data in 44.1kHz within [-1, 1].

    Returns
    -------
    None
    """
    try:
        sf.write(str(loc), x, Config.sample_rate, 'PCM_16')
    except Exception as e:
        logging.error(f"Error saving WAV file: {e}")


class Resampler:
    """
    A class for the UTAU resampling process.

    Attributes
    ----------
    in_file : str
        Path to input file.

    out_file : str
        Path to output file.

    pitch : str
        The pitch of the note.

    velocity : str or float
        The consonant velocity of the note.

    flags : str
        The flags of the note.

    offset : str or float
        The offset from the start for the render area of the sample.

    length : str or int
        The length of the stretched area in milliseconds.

    consonant : str or float
        The unstretched area of the render.

    cutoff : str or float
        The cutoff from the end or from the offset for the render area of the sample.

    volume : str or float
        The volume of the note in percentage.

    modulation : str or float
        The modulation of the note in percentage.

    tempo : str
        The tempo of the note.

    pitch_string : str
        The UTAU pitchbend parameter.

    Methods
    -------    
    render(self):
        The rendering workflow. Immediately starts when class is initialized.

    get_features(self):
        Gets the WORLD features either from a cached file or generating it if it doesn't exist.

    generate_features(self, features_path):
        Generates WORLD features and saves it for later.

    resample(self, features):
        Renders a WAV file using the passed WORLD features.
    """
    def __init__(self, in_file, out_file, pitch, velocity, flags='', offset=0, length=1000, consonant=0, cutoff=0, volume=100, modulation=0, tempo='!100', pitch_string='AA'):
        """Initializes the renderer and immediately starts it.

        Parameters
        ---------
        in_file : str
            Path to input file.

        out_file : str
            Path to output file.

        pitch : str
            The pitch of the note.

        velocity : str or float
            The consonant velocity of the note.

        flags : str
            The flags of the note.

        offset : str or float
            The offset from the start for the render area of the sample.

        length : str or int
            The length of the stretched area in milliseconds.

        consonant : str or float
            The unstretched area of the render.

        cutoff : str or float
            The cutoff from the end or from the offset for the render area of the sample.

        volume : str or float
            The volume of the note in percentage.

        modulation : str or float
            The modulation of the note in percentage.

        tempo : str
            The tempo of the note.

        pitch_string : str
            The UTAU pitchbend parameter.
        """
        self.in_file = Path(in_file)
        self.out_file = out_file
        self.pitch = note_to_midi(pitch)
        self.velocity = float(velocity)
        self.flags = {k : int(v) if v else None for k, v in flag_re.findall(flags.replace('/', ''))}
        self.offset = float(offset)
        self.length = int(length)
        self.consonant = float(consonant)
        self.cutoff = float(cutoff)
        self.volume = float(volume)
        self.modulation = float(modulation)
        self.tempo = float(tempo[1:])
        self.pitchbend = pitch_string_to_cents(pitch_string)
        print('****pitchbend', self.pitchbend)

        self.render()
    
    def render(self):
        """The rendering workflow. Immediately starts when class is initialized.

        Parameters
        ----------
        None
        """
        features = self.get_features()
        self.resample(features)

    def get_features(self):
        """Gets the WORLD features either from a cached file or generating it if it doesn't exist.

        Parameters
        ----------
        None

        Returns
        -------
        features : dict
            A dictionary of the F0, MGC, BAP, and average F0.
        """
        # Setup cache path file
        fname = self.in_file.name
        features_path = self.in_file.with_suffix(cache_ext)
        logging.info(f'------Cache path: {features_path}')
        features = None

        if 'G' in self.flags.keys():
            logging.info('G flag exists. Forcing feature generation.')
            features = self.generate_features(features_path)
        elif os.path.exists(features_path):
            # Load if it exists
            logging.info(f'Reading {fname}{cache_ext}.')
            features = np.load(features_path)
        else:
            # Generate if not
            logging.info(f'{fname}{cache_ext} not found. Generating features.')
            features = self.generate_features(features_path)

        return features
    def generate_features(self, features_path):
        """Generates PC-NSF-hifigan features and saves it for later.

        Parameters
        ----------
        features_path : str or file
            The path for caching the features.

        Returns
        -------
        features : dict
            A dictionary of the MEL.
        """
        x = read_wav(self.in_file)
        logging.info('Generating F0.')

        mel_extract_hop_size = melAnalysis_extract_hop_size(torch.from_numpy(x).to(dtype=torch.float32).unsqueeze(0), 0, 1).squeeze()
        print('**********mel_extract_hop_size', mel_extract_hop_size.shape)
        mel_extract_hop_size = dynamic_range_compression_torch(mel_extract_hop_size).numpy()

    
        logging.info('Saving features.')
        
        features = {'mel_extract_hop_size' : mel_extract_hop_size}
        np.savez_compressed(features_path, **features)

        return features
    def resample(self, features):
        """
        Renders a WAV file using the passed WORLD features.

        Parameters
        ----------
        features : dict
            A dictionary of the F0, MGC, BAP, and average F0.
 
        Returns
        -------
        None
        """
        if self.out_file == 'nul':
            logging.info('Null output file. Skipping...')
            return
        
        mod = self.modulation / 100
        print("--------mod", mod)
        
        self.out_file = Path(self.out_file)
        wave = read_wav(Path(self.in_file))

        logging.info('hachiming')
        mel_extract_hop_size = features['mel_extract_hop_size']
        print('**********mel_extract_hop_size', mel_extract_hop_size.shape)

        thop_extract_hop_size = Config.extract_hop_size / Config.sample_rate
        thop = Config.hop_size / Config.sample_rate

        t_area_mel_extract_hop_size = np.arange(mel_extract_hop_size.shape[1]) * thop_extract_hop_size + thop_extract_hop_size / 2
        tatal_time = t_area_mel_extract_hop_size[-1] + thop_extract_hop_size/2

        offset = self.offset / 1000 # start time
        cutoff = self.cutoff / 1000 # end time
        start = offset
        print('offset', offset)
        print('cutoff', cutoff)

        if self.cutoff < 0: # deal with relative end time
            end = start - cutoff       #???
        else:
            end = tatal_time - cutoff
        con = start + self.consonant / 1000
        print('--------con', con)
        print('--------start', start)
        print('--------end', end)
        print('--------tatal_time', tatal_time)
        print('--------wav_len', len(wave)/44100)
        logging.info('Calculating timing.') 

        logging.info('Preparing interpolators.')
        # Make interpolators to render new areas
        mel_interp = interp.Akima1DInterpolator(t_area_mel_extract_hop_size, mel_extract_hop_size, axis=1)

        length_req = self.length / 1000
        print('--------length_req', length_req)
        stretch_length = end - con
        print('--------stretch_length', stretch_length)
        if stretch_length < length_req:
            print('stretch_length < length_req')
            scaling_ratio = length_req / stretch_length

            def stretch(t, con, scaling_ratio):
                return np.where(t < con, t, con + (t - con) / scaling_ratio)
            
            stretched_n_frames = (con + (tatal_time - con)*scaling_ratio) // thop + 1
            stretched_t_mel = np.arange(stretched_n_frames) * thop + thop / 2

            stretch_t_mel = np.clip(stretch(stretched_t_mel, con, scaling_ratio),0,t_area_mel_extract_hop_size[-1])

            mel_render = mel_interp(stretch_t_mel)
        else:
            print('stretch_length >= length_req, no stretching needed.')
            scaling_ratio = 1
            
            mel_render = melAnalysis(torch.from_numpy(wave).to(dtype=torch.float32).unsqueeze(0), 0, 1).squeeze()
            mel_render = dynamic_range_compression_torch(mel_render).numpy()
            
        t = np.arange(mel_render.shape[1]) * thop
        # Calculate pitch in MIDI note number terms
        pitch = self.pitchbend / 100 + self.pitch
        t_pitch = 60 * np.arange(len(pitch)) / (self.tempo * 96) + start
        pitch_interp = interp.Akima1DInterpolator(t_pitch, pitch)
        pitch_render = pitch_interp(np.clip(t, start, t_pitch[-1]))
        f0_render = midi_to_hz(pitch_render)
        print('f0_render', f0_render.shape)
        print('mel_render', mel_render.shape)

        # 在start左边的mel帧数
        start_left_mel_frames = (start - thop/2)//thop
        if start_left_mel_frames > 64:
            cut_left_mel_frames = start_left_mel_frames - 64
        else:
            cut_left_mel_frames = 0
        # 在length_req+con右边的mel帧数
        mel_f=mel_render.shape[1]
        end_right_mel_frames = mel_f - (length_req+con - thop/2)//thop
        if end_right_mel_frames > 64:
            cut_right_mel_frames = end_right_mel_frames - 64
        else:
            cut_right_mel_frames = 0
        # 截取mel帧

        mel_render = mel_render[:, int(cut_left_mel_frames):int(mel_f-cut_right_mel_frames)]

        f0_render = f0_render[int(cut_left_mel_frames):int(mel_f-cut_right_mel_frames)]

        wav_con = m2w(f0_render,mel_render)
        render = wav_con[int(start*Config.sample_rate-Config.hop_size*cut_left_mel_frames):int((length_req+con)*Config.sample_rate-Config.hop_size*cut_left_mel_frames)]
        save_wav(self.out_file, render)

def split_arguments(input_string):
    # Regular expression to match two file paths at the beginning
    otherargs = input_string.split(' ')[-11:]
    file_path_strings = ' '.join(input_string.split(' ')[:-11])
    print(file_path_strings)
    first_file, second_file = file_path_strings.split('.wav ')
    return [first_file+".wav", second_file] + otherargs

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print(self.requestline)
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        post_data_string = post_data.decode('utf-8')
        print("传输ed data: " + post_data_string)
        #try:
        sliced = split_arguments(post_data_string)
        print("------------sliced",sliced)
        Resampler(*sliced)
        '''
        except Exception as e:
            trcbk = traceback.format_exc()
            self.send_response(500)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"An error occurred.\n{trcbk}".encode('utf-8'))
        self.send_response(200)
        self.end_headers()
        '''
def run(server_class=HTTPServer, handler_class=RequestHandler, port=8572):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting http server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    logging.info(f'hachisampler {version}')
    try:
        run()
    except Exception as e:
        name = e.__class__.__name__
        if name == 'TypeError':
            logging.info(help_string)
        else:
            raise e
