# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:30:31 2022

@author: amarmore
"""

import numpy as np
import seg_as.model.errors as err
import sklearn.metrics.pairwise as pairwise_distances
import warnings

def switch_autosimilarity(an_array, similarity_type, gamma = None, normalise = True):
    """
    High-level function to find the autosimilarity of this matrix.
    
    Expects a matrix of shape (Bars, Time-Frequency).
    
    Computes it with different possible similarity function:
        - "cosine" for the cosine similarity, i.e. the normalised dot product:
        .. math::
            s_{x_i,x_j} = \\frac{\langle x_i, x_j \rangle}{||x_i|| ||x_j||}
        -"covariance" for a covariance similarity, 
        i.e. the dot product of centered features:
        .. math::
            s_{x_i,x_j} = \langle x_i - \hat{x}, x_j - \hat{x} \rangle
        -"rbf" for the Radial Basis Function similarity, 
        i.e. the exponent of the opposite of the euclidean distance between features:
        .. math::
            s_{x_i,x_j} = \\exp^{-\\gamma ||x_i - x_j||_2}
        The euclidean distance can be the distance between the normalised features.
        Gamma is a parameter.
        See rbf_kernel from scikit-learn for more details.
    
    Parameters
    ----------
    an_array : numpy array
        The array/matrix seen as array which autosimilarity os to compute.
        Expected to be of shape (Bars, Time-Frequency).
    similarity_type : string
        Either "cosine", "covariance" or "rbf".
        It represents the type of similarity to compute between features.
    gamma : positive float, optional
        The gamma parameter in the rbf function, only used for the "rbf" similarity.
        The default is None, meaning that it is computed as function of the standard deviation,
        see get_gamma_std() for more details.
    normalise : boolean, optional
        Whether features should be normalised or not. 
        Normalisation depends on the similarity function.
        The default is True.

    Returns
    -------
    numpy array
        Autosimilarity matrix of the input an_array.

    """
    if similarity_type == "cosine":
        return get_cosine_autosimilarity(an_array)#, normalise = normalise)
    elif similarity_type == "covariance":
        return get_covariance_autosimilarity(an_array, normalise = normalise)
    elif similarity_type == "rbf":
        return get_rbf_autosimilarity(an_array, gamma, normalise = normalise)
    else:
        raise err.InvalidArgumentValueException(f"Incorrect similarity type: {similarity_type}. Should be cosine, covariance or rbf.")
        
def l2_normalise_barwise(an_array):
    """
    Normalises the array barwise (i.e., in its first dimension) by the l_2 norm.

    Parameters
    ----------
    an_array : numpy array
        The array which needs to be normalised.

    Returns
    -------
    numpy array
        The normalised array.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide") # Avoiding to show the warning, as it's handled, not te confuse the user.
        an_array_T = an_array.T/np.linalg.norm(an_array, axis = 1)
        an_array_T = np.where(np.isnan(an_array_T), 1e-10, an_array_T) # Replace null lines, avoiding best-path retrieval to fail
    return an_array_T.T

def get_cosine_autosimilarity(an_array):#, normalise = True):
    """
    Computes the autosimilarity matrix, where the similarity function is the cosine.
    
    The cosine similarity function is the normalised dot product, i.e.:
    .. math::
        s_{x_i,x_j} = \\frac{\langle x_i, x_j \rangle}{||x_i|| ||x_j||}
    
    Parameters
    ----------
    an_array : numpy array
        The array/matrix seen as array which autosimilarity os to compute.
        Expected to be of shape (Bars, Time-Frequency).

    Returns
    -------
    numpy array
        The autosimilarity of this array, with the cosine similarity function.

    """
    if type(an_array) is list:
        this_array = np.array(an_array)
    else:
        this_array = an_array
    #if normalise:
    this_array = l2_normalise_barwise(this_array)
    return this_array@this_array.T

def get_covariance_autosimilarity(an_array, normalise = True):
    """
    Computes the autosimilarity matrix, where the similarity function is the covariance.
    
    The covariance similarity function corresponds to the dot product of centered features:
    .. math::
        s_{x_i,x_j} = \langle x_i - \hat{x}, x_j - \hat{x} \rangle

    Parameters
    ----------
    an_array : numpy array
        The array/matrix seen as array which autosimilarity os to compute.
    normalise : boolean, optional
        Whether features should be normalised or not. 
        Normalisation here means that each centered feature is normalised by its norm.
        The default is True.
        
    Returns
    -------
    numpy array
        The covariance autosimilarity of this array.

    """
    if type(an_array) is list:
        this_array = np.array(an_array)
    else:
        this_array = an_array
    this_array = this_array - this_array.mean(axis=0)
    if normalise:
        this_array = l2_normalise_barwise(this_array)
    return this_array@this_array.T

def get_rbf_autosimilarity(an_array, gamma = None, normalise = True):
    """
    Computes the autosimilarity matrix, where the similarity function is the Radial Basis Function (RBF).
    
    The RBF corresponds to the exponent of the opposite of the euclidean distance between features:
    .. math::
        s_{x_i,x_j} = \\exp^{-\\gamma ||x_i - x_j||_2}
        
    The RBF is computed via scikit-learn.

    Parameters
    ----------
    an_array : numpy array
        The array/matrix seen as array which autosimilarity os to compute.
    gamma : positive float, optional
        The gamma parameter in the rbf function.
        The default is None, meaning that it is computed as function of the standard deviation,
        see get_gamma_std() for more details.
    normalise : boolean, optional
        Whether features should be normalised or not. 
        Normalisation here means that the euclidean norm is computed between normalised vectors.
        The default is True.

    Returns
    -------
    numpy array
        The RBF autosimilarity of this array.

    """
    if type(an_array) is list:
        this_array = np.array(an_array)
    else:
        this_array = an_array
    if gamma == None:
        gamma = get_gamma_std(this_array, scaling_factor = 1, no_diag = True, normalise = normalise)
    if normalise:
        this_array = l2_normalise_barwise(this_array)
    return pairwise_distances.rbf_kernel(this_array, gamma = gamma)

def get_gamma_std(barwise_TF, scaling_factor = 1, no_diag = True, normalise = True):
    """
    Default value for the gamm in the RBF similarity function.
    
    This default value is a function of the standard deviation of the values, more experiments should be made, so TODO.

    Parameters
    ----------
    barwise_TF : TYPE
        DESCRIPTION.
    scaling_factor : TYPE, optional
        DESCRIPTION. The default is 1.
    no_diag : TYPE, optional
        DESCRIPTION. The default is True.
    normalise : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if normalise:
        barwise_TF = l2_normalise_barwise(barwise_TF)
    euc_dist = pairwise_distances.euclidean_distances(barwise_TF)
    if not no_diag:
        #return scaling_factor/(np.std(euc_dist)*barwise_TF.shape[0])
        return scaling_factor/(2*np.std(euc_dist))
    else:
        for i in range(len(euc_dist)):
            euc_dist[i,i] = float('NaN')
        #return scaling_factor/(np.nanstd(euc_dist)*barwise_TF.shape[0])
        return scaling_factor/(2*np.nanstd(euc_dist))