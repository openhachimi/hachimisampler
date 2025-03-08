# hachimisampler
 A brand new resampler for utau based on [pc-nsf-hifigan](https://github.com/openvpi/vocoders).
## Why it's named hachimisampler?
hachimisampler was based on [straycat resampler](https://github.com/UtaUtaUtau/straycat) but uses pc-nsf-hifigan instead of world and the hachimi(ハチミ) means cat in Japanese.
## how to use? 
1. Install python 3.10 then run the commoand
```
pip install numpy scipy resampy pyworld torch onnxruntime praat-parselmouth
```
 2. Download the `hachiserver.py` and 'pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx' put them on a folder and run the 'hachisampler.py'.
 3. Open the UTAU and change the resampler to the path of `hachimisampler.exe`.
