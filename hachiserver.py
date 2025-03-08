import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)
import sys
import os
import pyworld as world # Vocoder
import numpy as np # Numpy <3
from numba import njit, vectorize, float64, optional # JIT compilation stuff (and ufuncs)
import soundfile as sf # WAV read + write
import scipy.signal as signal # for filtering
import scipy.interpolate as interp # Interpolator for feats
import scipy.ndimage as ndimage
import resampy # Resampler (as in sampling rate stuff)
from pathlib import Path # path manipulation
import re
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
import onnxruntime
from waveAnalyzer import MelAnalysis, F0Analyzer,resample_align_curve
import dataclasses
import soundfile as sf
import numpy as np
import torch

version = '0.0.1-ccb'
help_string = '''usage: straycat in_file out_file pitch velocity [flags] [offset] [length] [consonant] [cutoff] [volume] [modulation] [tempo] [pitch_string]

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
default_fs = 44100 # UTAU only really likes 44.1khz
fft_size = 2048
cache_ext = '.hjm.npz' # cache file extension

# Giving it better range
f0_floor = 20.0 
f0_ceil = 1600

# Flags
flags = ['fe', 'fl', 'fo', 'fv', 'fp', 've', 'vo', 'g', 't', 'A', 'B', 'G', 'P', 'S', 'p', 'R', 'D', 'C', 'Z']
flag_re = '|'.join(flags)
flag_re = f'({flag_re})([+-]?\\d+)?'
flag_re = re.compile(flag_re)

@dataclasses.dataclass
class Config:
    sampling_rate: int = 44100
    win_size: int = 2048
    hop_size: int = 512
    n_mels: int = 128
    n_fft: int = 2048
    mel_fmin: float = 20.0
    mel_fmax: float = 16000.0
    f0_extractor: str = 'parselmouth'
    f0_min: float = 20.0
    f0_max: float = 1600
    vocoder_path: str = r"pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx"


melAnalysis = MelAnalysis(
    sampling_rate=Config.sampling_rate, 
    win_size=Config.win_size, 
    hop_size=Config.hop_size, 
    n_mels=Config.n_mels, 
    n_fft=Config.n_fft, 
    mel_fmin=Config.mel_fmin, 
    mel_fmax=Config.mel_fmax
    )

f0Analyzer = F0Analyzer(
    sampling_rate = Config.sampling_rate,
    f0_extractor = Config.f0_extractor,
    hop_size = Config.hop_size,
    f0_min = Config.f0_min,
    f0_max = Config.f0_max
    )

#mel to wave
def m2w(f0,mel):
    '''
    mel shape = (n_mels, n_frames*speed)
    f0 shape = (n_frames*speed,)
    vocoder_keyshift shape = (n_frames,)
    '''

    # 加载vocoder模型
    ort_session = onnxruntime.InferenceSession(Config.vocoder_path)

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


# Utility functions
@vectorize([float64(float64, float64, float64)], nopython=True)
def smoothstep(edge0, edge1, x):
    """Smoothstep function from GLSL that works with numpy arrays."""
    x = (x - edge0) / (edge1 - edge0)
    if x < 0:
        x = 0
    elif x > 1:
        x = 1
    return 3*x*x - 2*x*x*x

@vectorize([float64(float64, float64, float64)], nopython=True)
def clip(x, x_min, x_max):
    """Clips function. Faster than np.clip somehow"""
    if x < x_min:
        return x_min
    if x > x_max:
        return x_max
    return x

@vectorize([float64(float64, float64)], nopython=True)
def bias(x, a):
    """Element-wise Schlick bias function."""
    if a == 0:
        return 0
    if a == 1:
        return 1
    return x / ((1 / a - 2) * (1 - x) + 1)

def highpass(x, fs=44100, cutoff=3000, order=1):
    """Butterworth highpass with doubled order because of sosfiltfilt."""
    nyq = 0.5 * fs
    cut = cutoff / nyq
    sos = signal.butter(order, cut, btype='high', output='sos')
    return signal.sosfiltfilt(sos, x)

def lowpass(x, fs=44100, cutoff=16000, order=1):
    """Butterworth lowpass with doubled order because of sosfiltfilt."""
    nyq = 0.5 * fs
    cut = cutoff / nyq
    sos = signal.butter(order, cut, btype='low', output='sos')
    return signal.sosfiltfilt(sos, x)

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

##def hz_to_midi(x):
##    return 12 * np.log2(x / 440) + 69

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

    if fs != default_fs:
        x = resampy.resample(x, fs, default_fs)

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
        sf.write(str(loc), x, default_fs, 'PCM_16')
    except Exception as e:
        logging.error(f"Error saving WAV file: {e}")

# Processing WORLD things
@njit(float64(float64[:], optional(float64), optional(float64)))
def _jit_base_frq(f0, f0_min, f0_max):
    q = 0
    avg_frq = 0
    tally = 0
    N = len(f0)

    if f0_min is None:
        f0_min = f0_floor

    if f0_max is None:
        f0_max = f0_ceil
    
    for i in range(N):
        if f0[i] >= f0_min and f0[i] <= f0_max:
            if i < 1:
                q = f0[i+1] - f0[i]
            elif i == N - 1:
                q = f0[i] - f0[i-1]
            else:
                q = (f0[i+1] - f0[i-1]) / 2
            weight = 2 ** (-q * q)
            avg_frq += f0[i] * weight
            tally += weight

    if tally > 0:
        avg_frq /= tally
    return avg_frq

def base_frq(f0, f0_min=None, f0_max=None):
    """Get average F0 with a stronger bias on flatter areas. 

    Parameters
    ----------
    f0 : list or ndarray
        Array of F0 values.

    f0_min : float, optional
        Lower F0 limit.

    f0_max : float, optional
        Upper F0 limit.

    Returns
    -------
    float
        Average F0.
    """
    return _jit_base_frq(f0, f0_min, f0_max)

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
        """Generates WORLD features and saves it for later.

        Parameters
        ----------
        features_path : str or file
            The path for caching the features.

        Returns
        -------
        features : dict
            A dictionary of the F0, MGC, BAP, and average F0.
        """
        x = read_wav(self.in_file)
        logging.info('Generating F0.')

        mel = melAnalysis(torch.from_numpy(x).to(dtype=torch.float32), 0, 1).numpy()
        f0 = f0Analyzer(torch.from_numpy(x).to(dtype=torch.float32), n_frames=mel.shape[1])[0]
        base_f0 = base_frq(f0)       
        logging.info('Saving features.')
        
        features = {'base' : base_f0, 'f0' : f0, 'mel' : mel}
        np.savez_compressed(features_path, **features)

        if len(x)//Config.hop_size != mel.shape[1]:
            logging.warning(f'Mel shape {mel.shape} does not match audio length {len(x)//Config.hop_size}.')

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
        
        self.out_file = Path(self.out_file)
        wave = read_wav(Path(self.in_file))

        logging.info('hachiming')
        base_f0 = features['base']
        f0 = features['f0']
        f0[f0 == 0] = base_f0
        f0_off = f0 - base_f0
        print('f0', f0.shape)
        mel = features['mel']

        thop = Config.hop_size / Config.sampling_rate

        t_area_f0 = np.arange(len(f0)) * thop
        t_area_mel = t_area_f0 + thop / 2
        tatal_time = t_area_f0[-1] + thop

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
        f0_off_interp = interp.UnivariateSpline(t_area_f0, f0_off, s=0, ext='const')
        mel_interp = interp.Akima1DInterpolator(t_area_mel, mel, axis=1)

        length_req = self.length / 1000
        print('--------length_req', length_req)
        stretch_length = end - con
        print('--------stretch_length', stretch_length)
        if stretch_length < length_req:
            print('stretch_length < length_req')
            speed = length_req / stretch_length

            def stretch(t, con, speed):
                return np.where(t < con, t, con + (t - con) / speed)
            
            stretched_n_frames = (con + (tatal_time - con)*speed) // thop + 1
            stretched_t_f0 = np.arange(stretched_n_frames) * thop
            stretched_t_mel = stretched_t_f0 + thop / 2

            print('stretched_t_f0', stretched_t_f0.shape)

            stretch_t_f0 = clip(stretch(stretched_t_f0, con, speed),0,t_area_f0[-1])
            stretch_t_mel = clip(stretch(stretched_t_mel, con, speed),0,t_area_mel[-1])

            print('stretch_t_f0', stretch_t_f0.shape)

            f0_off_render = f0_off_interp(stretch_t_f0)
            mel_render = mel_interp(stretch_t_mel)

            print('f0_off_render', f0_off_render.shape)
        else:
            print('stretch_length >= length_req, no stretching needed.')
            speed = 1
            f0_off_render = f0_off
            mel_render = mel
            
        t = np.arange(len(f0_off_render)) * thop
        # Calculate pitch in MIDI note number terms
        pitch = self.pitchbend / 100 + self.pitch
        t_pitch = 60 * np.arange(len(pitch)) / (self.tempo * 96) + start
        pitch_interp = interp.Akima1DInterpolator(t_pitch, pitch)
        pitch_render = pitch_interp(clip(t, start, t_pitch[-1]))
        f0_render = midi_to_hz(pitch_render) + f0_off_render * mod
        print('f0_render', f0_render.shape)
        print('mel_render', mel_render.shape)
        

        wav_con = m2w(f0_render,mel_render)
        render = wav_con[int(start*Config.sampling_rate):int(con*Config.sampling_rate + stretch_length*Config.sampling_rate*speed)]
        print('wav_con', wav_con.shape)
        print('start', int(start*Config.sampling_rate))
        print('con + stretch_length*Config.sampling_rate*speed', int(con*Config.sampling_rate + stretch_length*Config.sampling_rate*speed))
        print('leg', int((length_req+con)*Config.sampling_rate))
        print('render', render.shape)
        print('--------render_len', len(render)/44100)
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