## Usage

Preprocessing
```
python preprocess_audio.py
```

Training:
- train.py 실행 시 model/crn.py line 91, x = x.permute(0,1,3,2) 주석처리

```
python train.py -C config/train/baseline_model.json5

python train.py -C config/train/crn_baseline.json5
```

Inference:
- inference.py 실행 시 model/crn.py line 91, x = x.permute(0,1,3,2) 주석처리 제거

```
python inference.py -C config\inference\basic.json5 -cp Experiments\CRN\baseline_model\checkpoints\latest_model.tar -dist enhanced

python inference.py -C config\inference\basic.json5 -cp Experiments\CRN\crn_baseline\checkpoints\latest_model.tar -dist enhanced
```


## Environments

# platform: win-64
# conda

- python=3.9.0
- pytorch=2.3.1
- h5py
- json5
- librosa
- libsndfile
- matplotlib
- mkl
- numba
- numpy
- pesq
- pqdm
- pysoundfile
- pystoi
- scipy
- sox
- tensorflow=2.10.0
- torchaudio
- torchvision
- tqdm

## References

- [CRNN_mapping_baseline](https://github.com/YangYang/CRNN_mapping_baseline)
- [A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement](https://web.cse.ohio-state.edu/~wang.77/papers/Tan-Wang1.interspeech18.pdf)
- [EHNet](https://github.com/ododoyo/EHNet)
- [Convolutional-Recurrent Neural Networks for Speech Enhancement](https://arxiv.org/abs/1805.00579)
