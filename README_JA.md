[English](https://github.com/mtfotto/hachimisampler/blob/dingdongji/README.md) [正體中文](https://github.com/mtfotto/hachimisampler/blob/dingdongji/README_zh_tw.md) [简体中文](https://github.com/mtfotto/hachimisampler/blob/dingdongji/README_zh_cn.md)
# ハチミサンプルラー（hachimisampler)
ハチミサンプルラーは、 [pc-nsf-hifigan](https://github.com/openvpi/vocoders)をベースにしたUTAU用の全く新しいリサンプラーです。
## 使用方法は 
python 3.10をインストールして、以下のコマンドを実行する。（コンダを推奨）
```
pip install numpy scipy resampy pyworld torch onnxruntime praat-parselmouth
```
 2. [リリース](https://github.com/mtfotto/hachimisampler/releases)をダウンロードして解凍し、「hachiserver.py 」を実行してください。
 3. UTAUを開き、リサンプラーをhachimisampler.exeのパスに変更する。
# Thanks：
- [yjzxkxdn](https://github.com/yjzxkxdn)
- [openvpi](https://github.com/openvpi) for the pc-nsf-hifigan
