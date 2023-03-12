# Notes for applying Difusion-LM

This document summarizes the implementation/application notes for Diffusion-LM (Li et al., 2002).

**References**

Original paper: https://arxiv.org/abs/2205.14217

Original implementation code: https://github.com/XiangLi1999/Diffusion-LM

---

## The Diffusion Process



## The Model

The authors used `create_model_and_diffusion()` to create model.

https://github.com/XiangLi1999/Diffusion-LM/blob/main/improved-diffusion/improved_diffusion/script_util.py#L47

The model architecture (denoted as `model_arch`) can be `conv-unet`, `1d-unet`, `trans-unet`, `transformer` (used as the exmample on the authors' GitHub).

#### 'model_arch' == 'transformer'

UNet with attention and time encoding ([implimentation](https://github.com/XiangLi1999/Diffusion-LM/blob/759889d58ef38e2eed41a8c34db8032e072826f4/improved-diffusion/improved_diffusion/transformer_model2.py#L674)).

```python
## Excerpts from original code
# source: https://github.com/XiangLi1999/Diffusion-LM/blob/759889d58ef38e2eed41a8c34db8032e072826f4/improved-diffusion/improved_diffusion/script_util.py#L239
elif model_arch == 'transformer':
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    elif image_size == 16:  # DEBUG**
        channel_mult = (1, 2, 2, 2)
    else:
        channel_mult = (1, 2, 2, 2)

    attention_ds = []
    
    for res in attention_resolutions.split(","):   # attention_resolutions. defaut: "16,8" (?)
        attention_ds.append(image_size // int(res))
    
    # https://github.com/XiangLi1999/Diffusion-LM/blob/759889d58ef38e2eed41a8c34db8032e072826f4/improved-diffusion/improved_diffusion/transformer_model2.py#L674
    return TransformerNetModel2(
        in_channels=in_channel,  # 3, DEBUG**
        model_channels=num_channels,
        out_channels=(out_channel if not learn_sigma else out_channel*2),  # DEBUG**  (3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        config_name=config_name,
        training_mode=training_mode,
        vocab_size=vocab_size,
        experiment_mode=experiment_mode,
        logits_mode=logits_mode,
    )
```


## Training

Reference: https://github.com/XiangLi1999/Diffusion-LM#train-diffusion-lm

For the E2E task (50k restaurant reviews, 821 vocab), the embedding dimension of the tokenized input sequence is set to be **16**, modality to be **e2e-tgt**, submit to be **no**.

```sh
python scripts/run_train.py --diff_steps 2000 --model_arch transformer --lr 0.0001 --lr_anneal_steps 200000  --seed 102 --noise_schedule sqrt --in_channel 16 --modality e2e-tgt --submit no --padding_mode block --app "--predict_xstart True --training_mode e2e --vocab_size 821  --e2e_train ../datasets/e2e_data " --notes xstart_e2e
```

For the ROCStories task (98k five-sentence stories, 11k vocab), the embedding dimension of the tokenized input sequence is set to be **128**.

```sh
python scripts/run_train.py --diff_steps 2000 --model_arch transformer --lr 0.0001 --lr_anneal_steps 400000  --seed 101 --noise_schedule sqrt  --in_channel 128 --modality roc --submit no --padding_mode pad  --app "--predict_xstart True --training_mode e2e  --vocab_size 11043  --roc_train ../datasets/ROCstory " --notes xstart_e2e --bsz 64
```



## Decoding

### batch_decode.py

https://github.com/XiangLi1999/Diffusion-LM/blob/main/improved-diffusion/scripts/batch_decode.py

```
python scripts/batch_decode.py {path-to-diffusion-lm} -1.0 ema
```

The code runs the the commands:

1. Sample from the diffusion process

```
scripts/{mode}_sample.py --model_path {tgt} --batch_size 50 --num_samples {num_samples} --top_p {top_p} --out_dir {out_dir}
```

where {mode} can be `image` or `text`

2. Runs e2e-metrics/measure_scores.py 

3. Runs scripts/ppl_under_ar.py

### text_sample.py

https://github.com/XiangLi1999/Diffusion-LM/blob/main/improved-diffusion/scripts/text_sample.py

```python
## Excerpts from original code
# source: https://github.com/XiangLi1999/Diffusion-LM/blob/main/improved-diffusion/scripts/text_sample.py

def main():
    # parse arguments
    args = create_argparser().parse_args()
    
    # distributed process group
    dist_util.setup_dist()  # improved_diffusion/dist_util.py
    
    # OpenAI baselines
    logger.configure()  # improved_diffusion/logger.py

    # create model and diffusion
    # https://github.com/XiangLi1999/Diffusion-LM/blob/main/improved-diffusion/improved_diffusion/script_util.py#L47
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    # load weights
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    # conditional text generation
    if args.experiment_mode == 'conditional_gen':
    
        from improved_diffusion.text_datasets import load_data_text
        
        # load_models
        # https://github.com/XiangLi1999/Diffusion-LM/blob/main/improved-diffusion/improved_diffusion/rounding.py#L10
        model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                        os.path.split(args.model_path)[0])
        
        # mapping from token to word
        rev_tokenizer = {v: k for k, v in tokenizer.items()}
        
        # https://github.com/XiangLi1999/Diffusion-LM/blob/main/improved-diffusion/improved_diffusion/text_datasets.py#L18
        data = load_data_text(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            data_args=args,
            model=model2,
            deterministic=True,
            task_mode=args.modality,
            padding_mode=args.padding_mode,  # block, pad
            split=args.split,
            load_vocab=rev_tokenizer,
        )

    model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                    os.path.split(args.model_path)[0])
                                    
    if args.training_mode.startswith('e2e'):
        print('e2e, load the right model embeddings', '*'*80)
        model2.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())

    all_images = []
    all_labels = []
    model3 = get_weights(model2, args)
    
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.experiment_mode == 'conditional_gen':
            batch, model_kwargs = next(data)
            model_kwargs.pop('input_ids')
            if args.mbr_sample > 1:
                model_kwargs = {k: v.to(dist_util.dev()).repeat_interleave(args.mbr_sample, dim=0) for k, v in model_kwargs.items()}
            else:
                model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
            print([(k, v.shape) for (k,v) in model_kwargs.items()])
            
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
            
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        
        if args.model_arch == '1d-unet':
            if args.mbr_sample > 1 and args.experiment_mode == 'conditional_gen':
                sample_shape = (args.batch_size * args.mbr_sample, args.in_channel, args.image_size ** 2)
            else:
                sample_shape = (args.batch_size,  args.in_channel, args.image_size ** 2)
        else:
            if args.mbr_sample > 1 and args.experiment_mode == 'conditional_gen':
                sample_shape = (args.batch_size * args.mbr_sample, args.image_size ** 2, args.in_channel)
            else:
                sample_shape = (args.batch_size, args.image_size ** 2, args.in_channel)
        print(sample_shape)
        
        # sample x{t-1}
        sample = sample_fn(
            model,
            sample_shape,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model3.cuda()) if args.clamp == 'clamp' else None,
            model_kwargs=model_kwargs,
            top_p =args.top_p,
        )

        if args.model_arch == '1d-unet':
            sample = sample.permute(0, 2, 1)
            
        #
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        
        
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    print(arr.shape, 'full shape')
    arr = arr[: args.num_samples * args.mbr_sample]

    if diffusion.training_mode.startswith('e2e'):
        word_lst_e2e = []
        print('decoding for e2e', )
        print(arr.shape)
        x_t = th.tensor(arr).cuda()
        if args.model_arch == 'conv-unet':
            reshaped_x_t = x_t.view(x_t.size(0), -1, x_t.size(-1))
        else:
            reshaped_x_t = x_t
            
        logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
        
        cands = th.topk(logits, k=1, dim=-1)
        
        sample = cands.indices
        
        tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
        
        for seq in cands.indices:
            if isinstance(tokenizer, dict):
                tokens = " ".join([tokenizer[x[0].item()] for x in seq])
            else:
                tokens = tokenizer.decode(seq.squeeze(-1))
            word_lst_e2e.append(tokens)

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
        
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
        out_path = os.path.join(args.out_dir, f"{model_base_name}.samples_{args.top_p}.npz")
        # out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")

    if args.verbose == 'yes':
        logger.log('decode by rounding. ')
        print('load_models')
        if diffusion.training_mode.startswith('e2e'):
            word_lst = word_lst_e2e
        else:
            set_seed(101)
            model, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                           os.path.split(args.model_path)[0])
            print('rounding')
            
            # rounding
            # https://github.com/XiangLi1999/Diffusion-LM/blob/main/improved-diffusion/improved_diffusion/rounding.py#L78
            word_lst = rounding_func(args.experiment, arr, model, tokenizer,
                                     emb_scale_factor=args.emb_scale_factor)

        out_path2 = os.path.join(args.out_dir, f"{model_base_name}.samples_{args.top_p}.txt")
        fout = open(out_path2, 'w')

        for (xx) in zip( word_lst):
            # print('---' * 30)
            # print(tokenizer.decode(gg.tolist()))
            # print('xx' * 30)
            print(xx[0], file=fout)
            # print('---' * 30)
        fout.close()
        print(f'written the decoded output to {out_path2}')

        ##############
        out_path2 = os.path.join(args.out_dir, f"{model_base_name}.samples_{args.top_p}.json")
        fout = open(out_path2, 'w')
        for (xx) in zip(word_lst):
            print(json.dumps(xx), file=fout)
        fout.close()
        print(f'written the decoded output to {out_path2}')

```

### p_sample()

Location: `improved_diffusion/gaussian_diffusion.py`

https://github.com/XiangLi1999/Diffusion-LM/blob/759889d58ef38e2eed41a8c34db8032e072826f4/improved-diffusion/improved_diffusion/gaussian_diffusion.py

```python
## Excerpts from original code
# source: https://github.com/XiangLi1999/Diffusion-LM/blob/759889d58ef38e2eed41a8c34db8032e072826f4/improved-diffusion/improved_diffusion/gaussian_diffusion.py#L584

def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, top_p=None):

    """
    Sample x_{t-1} from the model at the given timestep.
    :param model: the model to sample from.
    :param x: the current tensor at x_{t-1}.
    :param t: the value of t, starting at 0 for the first diffusion step.
    :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
    :param denoised_fn: if not None, a function which applies to the
        x_start prediction before it is used to sample.
    :param model_kwargs: if not None, a dict of extra keyword arguments to
        pass to the model. This can be used for conditioning.
    :return: a dict containing the following keys:
             - 'sample': a random sample from the model.
             - 'pred_xstart': a prediction of x_0.
    """
    out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs)
    
    if top_p is not None and top_p > 0:
        noise = th.randn_like(x)  # normal distribution
        replace_mask = th.abs(noise) > top_p
        
        while replace_mask.any():
            noise[replace_mask] = th.randn_like(noise[replace_mask])
            replace_mask = th.abs(noise) > top_p
            
        assert (th.abs(noise) <= top_p).all()

    else:
        noise = th.randn_like(x)
        
    nonzero_mask = (
        (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
    )  # no noise when t == 0
    
    sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
    
    return {"sample": sample,
            "pred_xstart": out["pred_xstart"],
            "greedy_mean":out["mean"],
            "out":out}
```


### rounding_func()

Location: `improved_diffusion/rounding.py`

https://github.com/XiangLi1999/Diffusion-LM/blob/main/improved-diffusion/improved_diffusion/rounding.py

```python
## Excerpts from original code
# source: https://github.com/XiangLi1999/Diffusion-LM/blob/main/improved-diffusion/improved_diffusion/rounding.py#L78

def rounding_func(mode, text_emb_lst, model, tokenizer, emb_scale_factor=1.0):
    """
    
    Args:
        mode - 
        text_emb_lst - 
        model - 
        tokenizer - 
        emb_scale_factor - 
    """

    decoded_out_lst = []
    
    if mode in ['random', 'random_up_proj', 'glove']:
        down_proj_emb = model.weight  # input_embs
        down_proj_emb2 = None

        def get_knn(down_proj_emb, text_emb, dist='cos'):
            
            # calculate distance between down_proj_emb and text_emb
            if dist == 'cos':
                adjacency = down_proj_emb @ text_emb.transpose(1, 0).to(down_proj_emb.device)
            elif dist == 'l2':
                adjacency = down_proj_emb.unsqueeze(1).expand(-1, text_emb.size(0), -1) - text_emb.unsqueeze(0).expand(
                    down_proj_emb.size(0), -1, -1)
                adjacency = -torch.norm(adjacency, dim=-1)
                
            # return k largest elements along dimention 0
            topk_out = torch.topk(adjacency, k=6, dim=0)
            
            # return sorted k largest elements and their indices in adjacency
            return topk_out.values, topk_out.indices

        dist = 'l2'

        for text_emb in text_emb_lst:
            import torch
            text_emb = torch.tensor(text_emb)
            # print(text_emb.shape)
            if len(text_emb.shape) > 2:
                text_emb = text_emb.view(-1, text_emb.size(-1))
            else:
                text_emb = text_emb
            
            # calculate the closest neighbors between:
            # (cosine) None and text_emb
            # (L2) model.weight and text_emb
            val, indices = get_knn((down_proj_emb2 if dist == 'cos' else down_proj_emb),
                                   text_emb.to(down_proj_emb.device), dist=dist)
            
            # decode with the the closest element
            decoded_out = " ".join([tokenizer[i] for i in indices[0].tolist()])
            
            # 
            decoded_out_lst.append(decoded_out)

    return decoded_out_lst
```












