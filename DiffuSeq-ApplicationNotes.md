# Notes for applying DiffuSeq

**References**

Original paper: https://arxiv.org/abs/2210.08933

Original GitHub Repository: https://github.com/Shark-NLP/DiffuSeq

---

## Data Format

```python
#
data = load_data_text(...)  # diffuseq/text_datasets.py
```

**load_data_text()**

https://github.com/Shark-NLP/DiffuSeq/blob/main/diffuseq/text_datasets.py#L11





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

https://github.com/Shark-NLP/DiffuSeq/blob/main/sample_seq2seq.py



```python
# create model and diffusion
model, diffusion = create_model_and_diffusion(...) # basic_utils.py

"""
create_model_and_diffusion()
  
  model = TransformerNetModel(...)  # diffuseq/transformer_model

"""

```

```python
# initiate an instance of tokenizer
tokenizer = load_tokenizer(args)  # basic_utils.py


# create embedding layer (input dim: tokenizer.vocab_size, output dim: args.hidden_dim)
model_emb, tokenizer = load_model_emb(args, tokenizer) # basic_utils.py

```

**Decoding procedures**


```python
"""
Reference: https://github.com/Shark-NLP/DiffuSeq/blob/main/sample_seq2seq.py#L122
"""
```






