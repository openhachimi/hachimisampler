[English](https://github.com/mtfotto/hachimisampler/blob/dingdongji/README.md) [日本語](https://github.com/mtfotto/hachimisampler/blob/dingdongji/README_JA.md)
# 哈基踩（hachimisampler)
 一个基于 [pc-nsf-hifigan](https://github.com/openvpi/vocoders)的新的utau重采样器。
## 为什么叫哈基踩（hachimisampler）?
哈基踩由[straycat（流浪猫） resampler](https://github.com/UtaUtaUtau/straycat) 修改而来，用pc-nsf-hifigan替换了原来的world，哈基米 (ハチミ) 在日语里面是蜂蜜的意思。
## 如何使用? 
1. 安装python3.10并运行下面的指令（墙裂建议使用conda以方便管理环境）
```
pip install numpy scipy resampy onnxruntime soundfile pyloudnorm
```
2. 在torch官网下载cuda版本的pytorch (如果你确定只使用onnx版，那么可以下载cpu版的pytorch)
3. 下载 [release](https://github.com/mtfotto/hachimisampler/releases) 解压后运行 'hachisampler.py'.
4. 将utau的重采样器设置为 `hachimisampler.exe`.
# 感谢：
- [yjzxkxdn](https://github.com/yjzxkxdn)
- [openvpi](https://github.com/openvpi) for the pc-nsf-hifigan
