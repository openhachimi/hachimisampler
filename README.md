[简体中文](https://github.com/mtfotto/hachimisampler/blob/dingdongji/README_zh_cn.md) [日本語](https://github.com/mtfotto/hachimisampler/blob/dingdongji/README_JA.md)
# 🐱ハチミサンプルラー🐱
# hachimisampler
 A brand new resampler for utau based on [pc-nsf-hifigan](https://github.com/openvpi/vocoders).
## Why it's named hachimisampler?
hachimisampler was based on [straycat resampler](https://github.com/UtaUtaUtau/straycat) but uses pc-nsf-hifigan instead of world and the hachimi(ハチミ) means cat in Japanese.
注意：hachimisampler最初用于制作哈基米类人力vocaloid，但是最近的几次更新使它能够调正常utau虚拟歌手，考虑到命名问题我们决定开一个调utau虚拟歌手的新仓库，叫hifisampler。hachimisampler将专注于人力vocaloid 添加了如原声大碟这样的功能 请根据你的需求选用
## how to use? 
1. Install python 3.10 then run the commoand
```
pip install numpy scipy resampy pyworld torch onnxruntime praat-parselmouth soundfile pyloudnorm
```
2. Download the CUDA version of PyTorch from the official website. (If you are sure that you will only use the ONNX version, then you can download the CPU version of PyTorch)
3. Download the [release](https://github.com/mtfotto/hachimisampler/releases) unpack it and run the 'hachisampler.py'.
4. Open the UTAU and change the resampler to the path of `hachimisampler.exe`.
# Thanks
- [yjzxkxdn](https://github.com/yjzxkxdn)
- [openvpi](https://github.com/openvpi) for the pc-nsf-hifigan
