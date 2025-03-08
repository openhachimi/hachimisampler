# hachimisampler
 A brand new resampler for utau based on [pc-nsf-hifigan](https://github.com/openvpi/vocoders).
## Why it's named hachimisampler?
hachimisampler was based on [straycat resampler](https://github.com/UtaUtaUtau/straycat) but uses pc-nsf-hifigan instead of world and the hachimi(ハチミ) means cat in Japanese.
## how to use? 
1. You need to have Python installed. This was made using Python 3.10
```
pip install numpy scipy resampy pyworld torch onnxruntime praat-parselmouth
```
 2. Download the `hachiserver.py` file and run it.
 3. Open the UTAU and change the resampler to the path of `hachimisampler.exe`.
