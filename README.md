[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2502.04128)
 
# X-Codec-2.0

Paper: LLaSA: Scaling Train-time and Inference-time Compute for LLaMA-based Speech Synthesis, originally from https://huggingface.co/HKUSTAudio/xcodec2

We scale up to 24k sample rate, also add 25 TPS.

## Training

1. Use virtual environment,

```bash
python3 -m venv xcodec
./xcodec/bin/pip3 install -r requirements.txt
```

2. Prepare dataset,

```bash
./xcodec/bin/python3 get_tsv.py
```

Or you can prepare yourself the data, example, [example.txt],

```text
/path/audio1.wav
/path/audio2.wav
```

3. Download checkpoint, optional,

```bash
wget https://huggingface.co/HKUSTAudio/xcodec2/resolve/main/ckpt/epoch%3D4-step%3D1400000.ckpt
```

4. Interpolate to support 24k, [interpolate.ipynb](interpolate.ipynb).

5. Run finetune,

```bash
CUDA_VISIBLE_DEVICES="0,1" \
python3 train.py log_dir=24k \
train.trainer.devices=2 \
dataset.train.filelist="/home/husein/ssd3/gemma3/audio-files.txt" \
dataset.train.batch_size=18 \
dataset.val.filelist="/home/husein/ssd3/gemma3/audio-files-test.txt"
```