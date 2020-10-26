#! /usr/bin/env python3
# coding=utf-8
# Copyright 2018 The Uber AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command with bag of words:
python examples/run_pplm.py -B space --cond_text "The president" --length 100 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.01 --window_length 5 --kl_scale 0.01 --gm_scale 0.95

Example command with discriminator:
python examples/run_pplm.py -D sentiment --class_label 3 --cond_text "The lake" --length 10 --gamma 1.0 --num_iterations 30 --num_samples 10 --stepsize 0.01 --kl_scale 0.01 --gm_scale 0.95
"""

import argparse
import json
from operator import add
from typing import List, Optional, Tuple, Union
import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange
from tqdm import tqdm
from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter

import logging
from pplm_classification_head import ClassificationHead
import os
import sys

PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
BIG_CONST = 1e10

QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3
VERBOSITY_LEVELS = {
    'quiet': QUIET,
    'regular': REGULAR,
    'verbose': VERBOSE,
    'very_verbose': VERY_VERBOSE,
}

logger = logging.getLogger(__name__)


def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)


def perturb_past(
        past,
        model,
        last,
        decoder_input_ids,
        encoder_outs=None,
        unpert_past=None,
        unpert_logits=None,
        accumulated_hidden=None,
        grad_norms=None,
        stepsize=0.01,
        one_hot_bows_vectors=None,
        loss_type=0,
        num_iterations=3,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        kl_scale=0.01,
        device='cuda',
        verbosity_level=REGULAR
):
    # Generate inital perturbed past
    grad_accumulator = [
        (np.zeros(p.shape).astype("float32"))
        for p in past
    ]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(
            0.,
            1.0 + SMALL_CONST,
            1.0 / (window_length)
        )[1:]
    else:
        decay_mask = 1.0

    # TODO fix this comment (SUMANTH)
    # Generate a mask is gradient perturbated is based on a past window
    _, _, _, curr_length, _ = past[0].shape

    if curr_length > window_length and window_length > 0:
        window_mask = []
        for p_ in past:
            _, _, _, curr_length2, _ = p_.shape
            ones_key_val_shape = (
                    tuple(p_.shape[:-2])
                    + tuple([window_length])
                    + tuple(p_.shape[-1:])
            )

            zeros_key_val_shape = (
                    tuple(p_.shape[:-2])
                    + tuple([curr_length2 - window_length])
                    + tuple(p_.shape[-1:])
            )

            ones_mask = torch.ones(ones_key_val_shape)
            ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
            ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

            window_mask.append(torch.cat(
                (ones_mask, torch.zeros(zeros_key_val_shape)),
                dim=-2
            ).to(device))
    else:
        window_mask = []
        for p_ in past:
            window_mask.append(torch.ones_like(p_).to(device))
    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None

    for i in range(num_iterations):
        if verbosity_level >= VERBOSE:
            print("Iteration ", i + 1)
        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in grad_accumulator
        ]

        # Compute hidden using perturbed past
        perturbed_past = list(map(add, past, curr_perturbation))
        _, _, _, curr_length, _ = curr_perturbation[0].shape

        decoder_cached_states_perturbed_past = ()
        tmp = {}
        for i, p in enumerate(perturbed_past):
            prev_key, prev_value = p[0], p[1]
            if i % 2 == 0:
                tmp["MultiheadAttention." + str(i + 1) + ".attn_state"] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
            else:
                tmp["MultiheadAttention." + str(i + 1) + ".attn_state"] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                decoder_cached_states_perturbed_past += (tmp,)
                tmp = {}
        
        all_logits, _, all_hidden, _ = model.decoder(last, encoder_outs,
                                                incremental_state=decoder_cached_states_perturbed_past, temperature=0.9,)
        hidden = all_hidden[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(
            hidden,
            dim=1
        ).detach()
        # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []
        if loss_type == PPLM_BOW or loss_type == PPLM_BOW_DISCRIM:
            for one_hot_bow in one_hot_bows_vectors:
                bow_logits = torch.mm(probs, torch.t(one_hot_bow))
                bow_loss = -torch.log(torch.sum(bow_logits))
                loss += bow_loss
                loss_list.append(bow_loss)
            if verbosity_level >= VERY_VERBOSE:
                print(" pplm_bow_loss:", loss.data.cpu().numpy())
        
        kl_loss = 0.0
        if kl_scale > 0.0:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = (
                    unpert_probs + SMALL_CONST *
                    (unpert_probs <= SMALL_CONST).float().to(device).detach()
            )
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(
                device).detach()
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * (
                (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
            )
            if verbosity_level >= VERY_VERBOSE:
                print(' kl_loss', kl_loss.data.cpu().numpy())
            loss += kl_loss

        loss_per_iter.append(loss.data.cpu().numpy())
        if verbosity_level >= VERBOSE:
            print(' pplm_loss', (loss - kl_loss).data.cpu().numpy())

        # compute gradients
        loss.backward(retain_graph=True)

        # calculate gradient norms
        if grad_norms is not None and loss_type == PPLM_BOW:
            grad_norms = [
                torch.max(grad_norms[index], torch.norm(p_.grad * window_mask[index]))
                for index, p_ in enumerate(curr_perturbation)
            ]
        else:
            grad_norms = [
                (torch.norm(p_.grad * window_mask[index]) + SMALL_CONST)
                for index, p_ in enumerate(curr_perturbation)
            ]

        # normalize gradients
        grad = [
            -stepsize *
            (p_.grad * window_mask[index] / grad_norms[
                index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [
        to_var(torch.from_numpy(p_), requires_grad=True, device=device)
        for p_ in grad_accumulator
    ]
    pert_past = list(map(add, past, grad_accumulator))

    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter

def get_bag_of_words_indices(bag_of_words_ids_or_paths: List[str], tokenizer) -> \
        List[List[List[int]]]:
    bow_indices = []
    for id_or_path in bag_of_words_ids_or_paths:
        filepath = id_or_path

        with open(filepath, "r", encoding="utf-8-sig") as f:
            words = f.read().strip().split("\n")
        bow_indices.append(
            [tokenizer.encode_line(word.strip(), add_if_not_exist=False)[:-1]
             for word in words])

    return bow_indices


def build_bows_one_hot_vectors(bow_indices,  device='cuda'):
    if bow_indices is None:
        return None

    one_hot_bows_vectors = []
    for single_bow in bow_indices:
        single_bow_ = []
        for i in single_bow:
            for j in i:
                single_bow_.append([j])

        single_bow = single_bow_
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        one_hot_bow = torch.zeros(num_words, 50004).to(device)
        one_hot_bow.scatter_(1, single_bow.long(), 1)
        one_hot_bows_vectors.append(one_hot_bow)
    return one_hot_bows_vectors


def full_text_generation(
        model,
        tokenizer,
        generator=None,
        num_samples=1,
        src_text=None,
        device="cuda",
        bag_of_words=None,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR,
        **kwargs
):

    bow_indices = get_bag_of_words_indices(bag_of_words.split(";"),tokenizer)
    loss_type = PPLM_BOW


    unpert_gen_tok_text, _, _, unpert_gen_tok_tensor = generate_text_pplm(
        model=model,
        tokenizer=tokenizer,
        generator=generator,
        src_text=src_text,
        device=device,
        length=length,
        perturb=False,
        verbosity_level=verbosity_level
    )
    if device == 'cuda':
        torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []

    for i in range(num_samples):
        pert_gen_tok_text, discrim_loss, loss_in_time, pert_gen_tok_tensor = generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            generator=generator,
            src_text=src_text,
            device=device,
            perturb=True,
            bow_indices=bow_indices,
            loss_type=loss_type,
            length=length,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale,
            verbosity_level=verbosity_level
        )
        pert_gen_tok_texts.append(pert_gen_tok_text)
        losses_in_time.append(loss_in_time)

    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return unpert_gen_tok_tensor, pert_gen_tok_tensor, discrim_losses, losses_in_time


def generate_text_pplm(
        model,
        tokenizer,
        generator=None,
        past=None,
        src_text=None,
        device="cuda",
        perturb=True,
        bow_indices=None,
        loss_type=0,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR
):
    output_so_far = None
    if src_text is not None:
        output_so_far = {'src_tokens': src_text.unsqueeze(0).to(device),
                         'src_lengths': torch.tensor([len(src_text)]).to(device)}
    '''
    src_tokens = output_so_far['src_tokens']
    src_lengths = (src_tokens.ne(generator.eos) & src_tokens.ne(generator.pad)).long().sum(dim=1)
    input_size = src_tokens.size()
    bsz = input_size[0]
    src_len = input_size[1]
    beam_size = generator.beam_size
    '''

    # collect one hot vectors for bags of words
    one_hot_bows_vectors = build_bows_one_hot_vectors(bow_indices, device)

    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []

    if verbosity_level >= VERBOSE:
        range_func = trange(length, ascii=True)
    else:
        range_func = range(length)

    #encoder output compute
    encoder_outs = model.encoder(**output_so_far)

    decoder_input_ids = torch.tensor([50002]).unsqueeze(0).type(torch.LongTensor).to(device)

    result = []
    for i in range_func:
        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current_token

        # run model forward to obtain unperturbed
        if past is None and decoder_input_ids is not None:
            last = decoder_input_ids
            if decoder_input_ids.shape[1] >= 1:
                _, past, _, _ = model.decoder(decoder_input_ids, encoder_outs, temperature=0.9,)
                past_dict = ()
                for i, p in enumerate(past):
                    saved_state_self = p["MultiheadAttention."+ str(i*2+1) + ".attn_state"]
                    saved_state_encoder_decoder = p["MultiheadAttention."+ str(i*2+2) + ".attn_state"]

                    tmp = torch.stack((saved_state_self["prev_key"], saved_state_self["prev_value"]))
                    past_dict += (tmp,)

                    tmp = torch.stack((saved_state_encoder_decoder["prev_key"],
                                       saved_state_encoder_decoder["prev_value"]))
                    past_dict += (tmp,)

        # BARTmodel input encoder source
        unpert_logits, unpert_past, unpert_all_hidden, _ = model.decoder(decoder_input_ids, encoder_outs, temperature=0.9,)
        unpert_past_dict = ()
        for i, p in enumerate(unpert_past):
            saved_state_self = p["MultiheadAttention." + str(i * 2 + 1) + ".attn_state"]
            saved_state_encoder_decoder = p["MultiheadAttention." + str(i * 2 + 2) + ".attn_state"]

            tmp = torch.stack((saved_state_self["prev_key"], saved_state_self["prev_value"]))
            unpert_past_dict = unpert_past_dict + (tmp,)

            tmp = torch.stack((saved_state_encoder_decoder["prev_key"], saved_state_encoder_decoder["prev_value"]))
            unpert_past_dict = unpert_past_dict + (tmp,)

        unpert_last_hidden = unpert_all_hidden[-1]

        # check if we are abowe grad max length
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary
        if not perturb or num_iterations == 0:
            pert_past = past_dict

        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            if past_dict is not None:
                pert_past, _, grad_norms, loss_this_iter = perturb_past(
                    past_dict,
                    model,
                    last,
                    decoder_input_ids,
                    encoder_outs=encoder_outs,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    one_hot_bows_vectors=one_hot_bows_vectors,
                    loss_type=loss_type,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    gamma=gamma,
                    kl_scale=kl_scale,
                    device=device,
                    verbosity_level=verbosity_level
                )
                loss_in_time.append(loss_this_iter)
            else:
                pert_past = past_dict

        decoder_cached_states_past = ()
        tmp = {}
        for i, p in enumerate(pert_past):
            prev_key, prev_value = p[0], p[1]
            if i % 2 == 0:
                tmp["MultiheadAttention." + str(i+1) + ".attn_state"] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
            else:
                tmp["MultiheadAttention." + str(i+1) + ".attn_state"] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                decoder_cached_states_past += (tmp,)
                tmp = {}

        pert_logits, past, pert_all_hidden, _ = model.decoder(decoder_input_ids, encoder_outs,
                                                         incremental_state=decoder_cached_states_past, temperature=0.9,)
        past_dict = ()
        for i, p in enumerate(past):
            saved_state_self = p["MultiheadAttention." + str(i * 2 + 1) + ".attn_state"]
            saved_state_encoder_decoder = p["MultiheadAttention." + str(i * 2 + 2) + ".attn_state"]

            tmp = torch.stack((saved_state_self["prev_key"], saved_state_self["prev_value"]))
            past_dict = past_dict + (tmp,)

            tmp = torch.stack((saved_state_encoder_decoder["prev_key"], saved_state_encoder_decoder["prev_value"]))
            past_dict = past_dict + (tmp,)

        pert_logits = pert_logits[:, -1, :] / temperature #+ SMALL_CONST
        pert_probs = F.softmax(pert_logits, dim=-1)
        
        #Computed banned token prob
        pad_idx = tokenizer.pad()
        unk_idx = tokenizer.unk()
        
        #BIG_CONST = 1e10
        inf = float('inf')
        pert_logits[:,pad_idx] = -inf
        unpert_logits[:,-1, pad_idx] = -inf

        pert_logits[:, unk_idx] = -BIG_CONST
        unpert_logits[:,-1, unk_idx] = -BIG_CONST
        
        #banned fiv-gram token  
        if len(result) > 15:
            ngram_tok = result[-15:]
            for tok_idx in ngram_tok:
                pert_logits[:, tok_idx] = -inf
                unpert_logits[:, -1, tok_idx] = -inf
        else:
            ngram_tok = result
            if len(ngram_tok) == 0:
                pass
            else:
                for tok_idx in ngram_tok:
                    pert_logits[:, tok_idx] = -inf
                    unpert_logits[:, -1, tok_idx] = -inf

        # Fuse the modified model and original model
        if perturb:
            unpert_probs = F.softmax(unpert_logits[:,-1,:], dim=-1)

            pert_probs = ((pert_probs ** gm_scale) * (
                    unpert_probs ** (1 - gm_scale)))  # + SMALL_CONST
            pert_probs = top_k_filter(pert_probs, k=top_k,
                                      probs=True)  # + SMALL_CONST

            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

        else:
            pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
            pert_probs = F.softmax(pert_logits, dim=-1)

        # sample or greedy
        _, last = torch.topk(pert_probs, k=1, dim=-1)

        # update context/output_so_far appending the new token
        """output_so_far = (
            last if output_so_far is None
            else torch.cat((output_so_far, last), dim=1)
        )"""
        decoder_input_ids = (
            last if decoder_input_ids is None
            else torch.cat((decoder_input_ids, last), dim=1)
        )
        result.append(last.item())
        
         

        if last.item() == generator.eos:
            break

    return result, unpert_discrim_loss, loss_in_time, decoder_input_ids

def run_pplm_example(
        src_text="",
        num_samples=1,
        bag_of_words=None,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        seed=0,
        no_cuda=False,
        verbosity='regular',
        args=None
):
    # set Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set verbosiry
    verbosity_level = VERBOSITY_LEVELS.get(verbosity.lower(), REGULAR)

    # set the device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"


    utils.import_user_module(args)

    use_cuda = torch.cuda.is_available() and not args.cpu
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    state = checkpoint_utils.load_checkpoint_to_cpu(args.path, eval(args.model_overrides))
    args2 = state['args']

    # model load
    model = task.build_model(args2)
    model.load_state_dict(state['model'], strict=True)
    model.make_generation_fast_(
        beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
        need_attn=args.print_alignment,
    )
    if use_cuda:
        model.to(device)

    # Initialize generator
    generator = task.build_generator(args)

    # src_text tokenizing
    data_path = args.data_dir
    topic = data_path.split('/')[-1]
    topic = topic.split('.')[0]
    result_path = os.path.join(args.output_dir, topic)
    f1 = open(result_path + "_output.txt", 'w', encoding='utf8')
    for src in tqdm(src_text):
        tokenized_src = src_dict.encode_line(src, add_if_not_exist=False).type(torch.LongTensor)
        if tokenized_src.size()[0] > 512:
            tokenized_src = torch.cat([tokenized_src[:511], torch.tensor([50002])]) 
            #print(tokenized_src.size())

        unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
            model=model,
            tokenizer=tgt_dict,
            generator=generator,
            num_samples=num_samples,
            src_text=tokenized_src,
            device=device,
            bag_of_words=bag_of_words,
            length=length,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale,
            verbosity_level=verbosity_level
        )
        # untokenize unperturbed text
        '''
        bow_word_ids = set()
        if bag_of_words and colorama:
            bow_indices = get_bag_of_words_indices(bag_of_words.split(";"),
                                                   tgt_dict)
            for single_bow_list in bow_indices:
                # filtering all words in the list composed of more than 1 token
                filtered = list(filter(lambda x: len(x) <= 1, single_bow_list))
                # w[0] because we are sure w has only 1 item because previous fitler
                bow_word_ids.update(w[0] for w in filtered)
        '''
        # iterate through the perturbed texts
        unpert_gen_text = tgt_dict.string(unpert_gen_tok_text, args.remove_bpe, escape_unk=True)
        print(unpert_gen_text.replace(' ','').replace('▁', ' '))

        pert_gen_text = tgt_dict.string(pert_gen_tok_texts, args.remove_bpe, escape_unk=True)
        print(pert_gen_text.replace(' ','').replace('▁', ' '))

        for text in pert_gen_text:
            f1.write(text.replace(' ','').replace('▁', ' '))
        f1.write('\t')
        for text in unpert_gen_text:
            f1.write(text.replace(' ','').replace('▁', ' '))
        f1.write('\n')

    return


if __name__ == '__main__':
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)

    if len(args.data_dir) < 2:
        print('have not data')
        run_pplm_example(**vars(args))
    else:
        print('have data')
        with open(args.data_dir, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print('file read finish')
        print('the number of line : {}'.format(len(lines)))
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)

        run_pplm_example(
            src_text=lines,
            num_samples=args.num_samples,
            bag_of_words=args.bag_of_words,
            length=args.length,
            stepsize=args.step_size,
            temperature=args.temperature_pplm,
            top_k=args.top_k,
            num_iterations=args.num_iterations,
            grad_length=args.grad_length,
            horizon_length=args.horizon_length,
            window_length=args.window_length,
            decay=args.decay,
            gamma=args.gamma,
            gm_scale=args.gm_scale,
            kl_scale=args.kl_scale,
            seed=args.seed,
            no_cuda=args.no_cuda,
            verbosity=args.verbosity,
            args=args
        )
