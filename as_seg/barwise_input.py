# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 18:34:29 2021

@author: amarmore
"""

import numpy as np
import as_seg.data_manipulation as dm
import as_seg.autosimilarity_segmentation as as_seg
import as_seg.model.errors as err
import tensorly as tl

# %% Tensor-spectrogram definition (careful: different mode organization than for NTD)
def tensorize_barwise_BFT(spectrogram, bars, hop_length_seconds, subdivision):
    """
    Returns a tensor-spectrogram from the original spectrogram and bars starts and ends.
    Each bar of the tensor-spectrogram will contain the same number of frames, define by the "subdivision" parameter.
    These frames are selected from an oversampled spectrogram, adapting to the specific size of each bar.

    Parameters
    ----------
    spectrogram : list of list of floats or numpy array
        The spectrogram to return as a tensor-spectrogram.
    bars : list of tuples
        List of the bars (start, end), in seconds, to cut the spectrogram at bar delimitation.
    hop_length_seconds : float
        The hop_length, in seconds.
    subdivision : integer
        The number of subdivision of the bar to be contained in each slice of the tensor.

    Returns
    -------
    np.array tensor
        The tensor-spectrogram as a np.array.

    """
    barwise_spec = []
    bars_idx = dm.segments_from_time_to_frame_idx(bars[1:], hop_length_seconds)
    for idx, beats in enumerate(bars_idx):
        t_0 = beats[0]
        t_1 = beats[1]
        samples = [int(round(t_0 + k * (t_1 - t_0)/subdivision)) for k in range(subdivision)]
        if samples[-1] < spectrogram.shape[1]:
            barwise_spec.append(spectrogram[:,samples])
    return np.array(barwise_spec)

def barwise_TF_matrix(spectrogram, bars, hop_length_seconds, subdivision):
    tensor_spectrogram = tensorize_barwise_BFT(spectrogram, bars, hop_length_seconds, subdivision)
    return tl.unfold(tensor_spectrogram, 0)

def vector_to_matrix(vector, freq_len, subdivision):
    return tl.fold(vector, 0, (freq_len,subdivision))

