[English](https://github.com/mtfotto/hachimisampler/blob/dingdongji/README.md) [正軆中文（台湾）](https://github.com/mtfotto/hachimisampler/blob/dingdongji/README_zh_tw.md) [日本語](https://github.com/mtfotto/hachimisampler/blob/dingdongji/README_JA.md)
# 哈基踩（hachimisampler)
 一个基于 [pc-nsf-hifigan](https://github.com/openvpi/vocoders)的新的utau重采样器。
## 为什么叫哈基踩（hachimisampler）?
哈基踩由[straycat（流浪猫） resampler](https://github.com/UtaUtaUtau/straycat) 修改而来，用pc-nsf-hifigan替换了原来的world，哈基米 (ハチミ) 在日语里面是小猫的意思。
## 如何使用? 
1. 安装python3.10并运行下面的指令（墙裂建议使用conda以方便管理环境）
```
pip install numpy scipy resampy pyworld torch onnxruntime praat-parselmouth
```
 2. 下载 [release](https://github.com/mtfotto/hachimisampler/releases) 解压后运行 'hachisampler.py'.
 3. 将utau的重采样器设置为 `hachimisampler.exe`.
# 感谢：
- [yjzxkxdn](https://github.com/yjzxkxdn)
- [openvpi](https://github.com/openvpi) for the pc-nsf-hifigan
