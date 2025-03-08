[简体中文](https://github.com/mtfotto/hachimisampler/blob/dingdongji/README_zh_cn.md) [正軆中文](https://github.com/mtfotto/hachimisampler/blob/dingdongji/README_zh_tw.md) [日本語](https://github.com/mtfotto/hachimisampler/blob/dingdongji/README_JA.md)
# 哈基踩（hachimisampler)
# hachimisampler
 A brand new resampler for utau based on [pc-nsf-hifigan](https://github.com/openvpi/vocoders).
## Why it's named hachimisampler?
hachimisampler was based on [straycat resampler](https://github.com/UtaUtaUtau/straycat) but uses pc-nsf-hifigan instead of world and the hachimi(ハチミ) means cat in Japanese.
## how to use? 
1. Install python 3.10 then run the commoand
```
pip install numpy scipy resampy pyworld torch onnxruntime praat-parselmouth
```
 2. Download the [release](https://github.com/mtfotto/hachimisampler/releases) unpack it and run the 'hachisampler.py'.
 3. Open the UTAU and change the resampler to the path of `hachimisampler.exe`.
# Thanks
- [yjzxkxdn](https://github.com/yjzxkxdn)
- [openvpi](https://github.com/openvpi) for the pc-nsf-hifigan
