[English](https://github.com/mtfotto/hachimisampler/blob/dingdongji/README.md) [简体中文](https://github.com/mtfotto/hachimisampler/blob/dingdongji/README_zh_cn.md)
# ハチミサンプルラー（hachimisampler)
ハチミサンプルラーは、 [pc-nsf-hifigan](https://github.com/openvpi/vocoders)をベースにしたUTAU用の全く新しいリサンプラーです。
## 使用方法
python 3.10をインストールして、以下のコマンドを実行する。（コンダを推奨）
```
pip install numpy scipy resampy pyworld torch onnxruntime praat-parselmouth soundfile pyloudnorm
```
2. 公式ウェブサイトからPyTorchのCUDA版をダウンロードしてください。（もしONNX版のみを使用することが確実であれば、CPU版のPyTorchをダウンロードすることができます）
3. [リリース](https://github.com/mtfotto/hachimisampler/releases)をダウンロードして解凍し、「hachiserver.py 」を実行してください。
4. UTAUを開き、リサンプラーをhachimisampler.exeのパスに変更する。
# 謝辞：
- [openvpi](https://github.com/openvpi) for the pc-nsf-hifigan
