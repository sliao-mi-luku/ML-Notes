# Notes for applying DiffuSeq

**References**

Original paper: https://arxiv.org/abs/2210.08933

Original GitHub Repository: https://github.com/Shark-NLP/DiffuSeq

---


## Decoding

**run_decode.sh**

https://github.com/Shark-NLP/DiffuSeq/blob/main/scripts/run_decode.sh

- model_dir (default: diffusion_models)
- seed (default: 123)
- split (default: test)

Runs `run_decode.py` to decode


**run_decode.py**

https://github.com/Shark-NLP/DiffuSeq/blob/main/scripts/run_decode.py

Argumets:

- model_dir
- seed
- step (default: 2000 | number of diffusion steps)
- bsz (default: 50 | batch size)
- split (default: 'test' | dataset for decoding)
- top_p (default: -1 (off) | top p for sampling)
- pattern (default: ema | training pattern)

Runs `sample_seq2seq.py` with arguments `model_path`, `step`, `batch_size`, `seed2`, `split`, `out_dir`, `top_p`.



**sample_seq2seq.py**

```python
# create model and diffusion
model, diffusion = create_model_and_diffusion(...)


```
