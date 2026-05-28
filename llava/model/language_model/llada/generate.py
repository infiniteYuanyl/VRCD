import torch
import numpy as np

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def get_num_transfer_tokens_sch(mask_index, steps,schedule=None,schedule_kwargs=None):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    if schedule is None:
        return get_num_transfer_tokens(mask_index,steps)
    if schedule_kwargs is None:
        schedule_kwargs = {}
   
    mask_num = mask_index.sum(dim=1, keepdim=True)
    steps = int(min(steps,mask_num[0]))
    t = torch.linspace(0, 1, steps+1)
    # at least one sample per step
    if schedule =='logit_normal':
      sigmas = sigmoid_normal_cdf(t)
    elif schedule =='shift':
      sigmas = logit_normal_schedule(schedule_kwargs.get('shift',3),t)
    elif schedule == 'cosine':
        sigmas = cosine_schedule(t)
    else:
      sigmas = t
    sigmas = sigmas.to(mask_num.device)
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
    
    for i in range(mask_num.size(0)):
      # print(sigmas.shape)
      sigmas_sample = (sigmas*mask_num[i]).to(torch.int64)
      # print(sigmas_sample)
      sigmas_sample = sigmas_sample[1:]-sigmas_sample[:-1]
      # print(sigmas_sample)
      # fix detal
      sigmas_sample = torch.clamp(sigmas_sample,1,None) # should only increase
      delta = sigmas_sample.sum() - mask_num[i]
    #   breakpoint()
      assert delta>=0
      j = 0
      
      while delta > 0:
        j = j % len(sigmas_sample) 
        if sigmas_sample[j] == 1:
          j += 1
          continue
        
        delta -= 1
        sigmas_sample[j] -= 1
        j += 1
    #   breakpoint()
      assert sigmas_sample.sum()==mask_num[i]
      num_transfer_tokens[i] = sigmas_sample#.to(torch.int64)
    return num_transfer_tokens.flip(-1)

def linear(y):
    return y

def cosine_schedule(x):
    """
    Cosine schedule mapping [0, 1] -> [1, 0]
    """
    x = np.clip(x, 0, 1)
    return 1-0.5 * (1 + np.cos(np.pi * x))

def sigmoid_normal_cdf(y):
    # y must be in (0, 1)
    logit_y = torch.log(y / (1 - y))
    return 0.5 * (1 + torch.erf(logit_y / torch.sqrt(torch.tensor(2.0))))
def logit_normal_schedule(shift,sigmas):
    # shift = 1 / shift
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    return sigmas
import os
DEBUG_PRINT_OUTPUT = os.environ.get('DEBUG_PRINT_OUTPUT',False)

from llava.model.language_model.llada.vrcd import compute_candidate_scores, normalize_decoding_method, select_transfer_positions

@ torch.no_grad()
def generate(model, prompt=None, steps=None, max_new_tokens=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336,inputs_embeds=None, position_ids=None,attention_mask=None,
              tokenizer=None,
                verbose=False,
                step_per_block=None,
                prefix_lm=False,
                schedule=None,
                schedule_kwargs=None,
                draft_tokens=None,
                step_ratio=None,
                method=None,
                decoding_method=None,
                vrcd_prefix_input_ids=None,
                vrcd_image_token_id=-200,
                vrcd_alpha=1.5,
                vrcd_lambda=2.0,
                vrcd_window_lambda=None,
             **kwargs):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Internal remasking strategy used by the LLaDA sampler.
        method: Internal decoding selector. The public inference path uses 'vrcd'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    # breakpoint()
    # remasking = 
    # step_ratio = 0.5
    # block_length = 1024
    # steps = 1024
    selected_method = normalize_decoding_method(decoding_method or method, remasking)
    if vrcd_window_lambda is not None:
        vrcd_lambda = vrcd_window_lambda
    steps = max_new_tokens # min(steps,max_new_tokens)
    # if step_ratio:
    #     steps = int(max_new_tokens*step_ratio)
    gen_length = max_new_tokens
    assert position_ids is None
    if prompt is None:
        assert inputs_embeds is not None
        bsz, seq_len = inputs_embeds.shape[:2]
        prompt = torch.full((bsz, seq_len), 0, dtype=torch.long).to(model.device)
    past_key_values = None
    if prefix_lm:
        past_key_values = model(None,input_embeddings=inputs_embeds,use_cache=True).attn_key_values
        # breakpoint()
        x = torch.full((bsz, gen_length), mask_id, dtype=torch.long).to(model.device)
        prompt = torch.full((bsz, 0), 0, dtype=torch.long).to(model.device)
        # x[:, :prompt.shape[1]] = prompt.clone()
    else:
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
        x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)
    # assert prompt.shape[0] == 1
    if draft_tokens is not None:
        assert draft_tokens.shape[1] <= gen_length
        x[:, prompt.shape[1]:prompt.shape[1]+draft_tokens.shape[1]] = draft_tokens.clone()

    # if block_length < gen_length:
    #    block_length = gen_length
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert ( steps % num_blocks == 0) or step_per_block is not None
    steps = steps // num_blocks
    if step_per_block:
        steps = min(step_per_block,block_length)
        assert step_ratio is None, 'Please do not pass both step_ratio and step_per_block'
    # step_ratio = 0.5
    # schedule = 'shift'
    # schedule_kwargs = dict(shift=3)
    # breakpoint()
    if step_ratio:
        steps = int(steps*step_ratio)

    # print(steps,step_per_block,block_length,draft_tokens.shape[-1])
    # NFE = 0
    if verbose:
        history = []
    for num_block in range(num_blocks):

        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens_sch(block_mask_index, steps,schedule=schedule,schedule_kwargs=schedule_kwargs)
        if DEBUG_PRINT_OUTPUT:
            print(f"Block: {num_block + 1}/{num_blocks}, Steps per Block: {steps}, Block Length: {block_length}")
            print(f"Tokens generated per step {num_transfer_tokens[0]}")
        for i in range(steps):
            # print(i)
            mask_index = (x == mask_id)
            block_mask_index = mask_index[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:]
            # print(mask_index.sum())
            if block_mask_index.sum() == 0:
                continue
            # NFE += 2
            if cfg_scale > 0.:
                assert NotImplementedError('cfg_scale > 0. is not supported.')
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                #
                logits = model(x_,input_embeds_inference=[inputs_embeds,None]).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                inputs_embeds_curr = model.transformer.wte(x)
                #print(tokenizer.batch_decode(x)[0].replace('<|endoftext|>',''))
                # print((x==mask_id).sum())
                # breakpoint()
                need_hidden_states = selected_method == "vrcd"
                if prefix_lm:
                    # breakpoint()
                    model_out = model(None,input_embeddings=inputs_embeds_curr,past_key_values=past_key_values,output_hidden_states=need_hidden_states)
                else:
                    if inputs_embeds is not None:
                        inputs_embeds_curr[:,:inputs_embeds.shape[1]] = inputs_embeds
                    model_out = model(None,input_embeddings=inputs_embeds_curr,output_hidden_states=need_hidden_states)
                logits = model_out.logits
            block_end = prompt.shape[1] + (num_block + 1) * block_length
            x0, confidence = compute_candidate_scores(
                logits,
                x_current=x,
                mask_index=mask_index,
                method=selected_method,
                temperature=temperature,
                block_end=block_end,
            )

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                select_index = select_transfer_positions(
                    score_row=confidence[j],
                    select_k=int(num_transfer_tokens[j, i].item()),
                    method=selected_method,
                    base_model=model,
                    hidden_states=getattr(model_out, "hidden_states", None) if cfg_scale <= 0. else None,
                    past_key_values=past_key_values,
                    prefix_input_ids=vrcd_prefix_input_ids,
                    row_idx=j,
                    image_token_id=int(vrcd_image_token_id),
                    vrcd_alpha=float(vrcd_alpha),
                    vrcd_lambda=float(vrcd_lambda),
                )
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            if verbose:
                history.append(x.clone().cpu())
    # breakpoint()
    # print(f"NFE: {NFE} Num Blocks: {num_blocks}")
    if verbose:
        return x,history
    return x
