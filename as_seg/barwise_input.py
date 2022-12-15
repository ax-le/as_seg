# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 18:34:29 2021

@author: amarmore

Module used to handle compute the Barwise TF matrix, presented in [1]
(Barwise TF matrix: a 2D representation of barwise features, 
each feature representing Time-Frequency content, where time is expressed at barscale)

References
----------

[1] A. Marmoret, J.E. Cohen, and F. Bimbot, "Barwise Compression
Schemes for Audio-Based Music Structure Analysis", in 19th Sound and Music
Computing Conference, SMC 2022, Sound and music Computing network, 2022.
"""

import as_seg.data_manipulation as dm
import as_seg.model.errors as err

import numpy as np
import tensorly as tl

# %% Spectrogram ordering modes as Bars-Frequency-Time 
#(careful: different mode organization than for NTD where it is Frequency-Time-Bars)
def tensorize_barwise_BFT(spectrogram, bars, hop_length_seconds, subdivision):
    """
    Returns a 3rd order tensor-spectrogram from the original spectrogram and bars starts and ends.
    
    Each bar in the tensor-spectrogram contains the same number of frames, define by the "subdivision" parameter.
    These frames are selected from an oversampled spectrogram, adapting to the specific size of each bar.
    See [1] for details.

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
    """
    Barwise TF matrix, a 2D representation of Barwise spectrograms as Time-Frequency vectors.
    See [1] for details.

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
    np.array
        The Barwise TF matrix, of sizes (b, tf).

    """
    tensor_spectrogram = tensorize_barwise_BFT(spectrogram, bars, hop_length_seconds, subdivision)
    return tl.unfold(tensor_spectrogram, 0)

def TF_vector_to_TF_matrix(vector, frequency_dimension, subdivision):
    """
    Encapsulating the conversion from a Time-Frequency vector to a Time-Frequency matrix (spectrogram)

    Parameters
    ----------
    vector : np.array
        A Time-Frequency vector (typically a row in the Barwise TF matrix).
    frequency_dimension : positive integer
        The size of the frequency dimension 
        (number of components in this dimension).
    subdivision : positive integer
        The size of the time dimension at the bar scale 
        (number of time components in each bar, defined as parameter when creating the Barwise TF matrix).

    Returns
    -------
    np.array
        A Time-Frequency matrix (spectrogram) of size (frequency_dimension, subdivision).

    """
    assert frequency_dimension*subdivision == vector.shape[0]
    return tl.fold(vector, 0, (frequency_dimension,subdivision))

