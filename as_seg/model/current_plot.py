# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:29:17 2019

@author: amarmore
"""

# Defining current plotting functions.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_me_this_spectrogram(spec, title = "Spectrogram", x_axis = "x_axis", y_axis = "y_axis", invert_y_axis = True, cmap = cm.Greys, figsize = None, norm = None, vmin = None, vmax = None):
    """
    Plots a spectrogram in a colormesh.
    """
    if figsize != None:
        plt.figure(figsize=figsize)
    elif spec.shape[0] == spec.shape[1]:
        plt.figure(figsize=(7,7))
    padded_spec = spec #pad_factor(spec)
    plt.pcolormesh(np.arange(padded_spec.shape[1]), np.arange(padded_spec.shape[0]), padded_spec, cmap=cmap, norm = norm, vmin = vmin, vmax = vmax, shading='auto')
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    if invert_y_axis:
        plt.gca().invert_yaxis()
    plt.show()
    
def pad_factor(factor):
    """
    Pads the factor with zeroes on both dimension.
    This is made because colormesh plots values as intervals (post and intervals problem),
    and so discards the last value.
    """
    padded = np.zeros((factor.shape[0] + 1, factor.shape[1] + 1))
    for i in range(factor.shape[0]):
        for j in range(factor.shape[1]):
            padded[i,j] = factor[i,j]
    return np.array(padded)
    
def compare_both_annotated(spec_1, spec_2, ends_1, ends_2, annotations, title_1 = "First", title_2 = "Second"):
    """
    Compare two Q matrices, with the annotation of the segmentation.
    Given two different Q, it will plot Q^T and Q.Q^T (autosimilairty of Q^T).
    """
    fig, axs = plt.subplots(1, 2, figsize=(20,10))
    padded_spec_1 = pad_factor(spec_1)
    axs[0].pcolormesh(np.arange(padded_spec_1.shape[0]), np.arange(padded_spec_1.shape[0]), padded_spec_1)
    axs[0].set_title('Autosimilarity of ' + title_1)
    padded_spec_2 = pad_factor(spec_2)
    axs[1].pcolormesh(np.arange(padded_spec_2.shape[0]), np.arange(padded_spec_2.shape[0]), padded_spec_2)
    axs[1].set_title('Autosimilarity of ' + title_2)
    axs[0].invert_yaxis()
    axs[1].invert_yaxis()    
    for x in annotations:
        axs[0].plot([x,x], [0,len(spec_1) - 1], '-', linewidth=1, color = "red")
        axs[1].plot([x,x], [0,len(spec_2) - 1], '-', linewidth=1, color = "red")
    for x in ends_1:
        if x in annotations:
            axs[0].plot([x,x], [0,len(spec_1)], '-', linewidth=1, color = "#8080FF")#"#17becf")
        else:
            axs[0].plot([x,x], [0,len(spec_1)], '-', linewidth=1, color = "orange")
    for x in ends_2:
        if x in annotations:
            axs[1].plot([x,x], [0,len(spec_2)], '-', linewidth=1, color = "#8080FF")#"#17becf")
        else:
            axs[1].plot([x,x], [0,len(spec_2)], '-', linewidth=1, color = "orange")
    plt.show()
    
def plot_measure_with_annotations(measure, annotations, color = "red"):
    """
    Plots the measure (typically novelty) with the segmentation annotation.
    """
    plt.plot(np.arange(len(measure)),measure, color = "black")
    for x in annotations:
        plt.plot([x, x], [0,np.amax(measure)], '-', linewidth=1, color = color)
    plt.show()
    
def plot_measure_with_annotations_and_prediction(measure, annotations, frontiers_predictions, title = "Title"):
    """
    Plots the measure (typically novelty) with the segmentation annotation and the estimated segmentation.
    """
    plt.title(title)
    plt.plot(np.arange(len(measure)),measure, color = "black")
    ax1 = plt.axes()
    ax1.axes.get_yaxis().set_visible(False)
    for x in annotations:
        plt.plot([x, x], [0,np.amax(measure)], '-', linewidth=1, color = "red")
    for x in frontiers_predictions:
        if x in annotations:
            plt.plot([x, x], [0,np.amax(measure)], '-', linewidth=1, color = "#8080FF")#"#17becf")
        else:
            plt.plot([x, x], [0,np.amax(measure)], '-', linewidth=1, color = "orange")
    plt.show()

def plot_spec_with_annotations(factor, annotations, color = "yellow", title = None):
    """
    Plots a spectrogram with the segmentation annotation.
    """
    if factor.shape[0] == factor.shape[1]:
        plt.figure(figsize=(7,7))
    plt.title(title)
    padded_fac = pad_factor(factor)
    plt.pcolormesh(np.arange(padded_fac.shape[1]), np.arange(padded_fac.shape[0]), padded_fac, cmap=cm.Greys)
    plt.gca().invert_yaxis()
    for x in annotations:
        plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = color)
    plt.show()
    
def plot_spec_with_annotations_abs_ord(factor, annotations, color = "green", title = None, cmap = cm.gray):
    """
    Plots a spectrogram with the segmentation annotation in both x and y axes.
    """
    if factor.shape[0] == factor.shape[1]:
        plt.figure(figsize=(7,7))
    plt.title(title)
    padded_fac = pad_factor(factor)
    plt.pcolormesh(np.arange(padded_fac.shape[1]), np.arange(padded_fac.shape[0]), padded_fac, cmap=cmap)
    plt.gca().invert_yaxis()
    for x in annotations:
        plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = color)
        plt.plot([0,len(factor)], [x,x], '-', linewidth=1, color = color)
    plt.show()

def plot_spec_with_annotations_and_prediction(factor, annotations, predicted_ends, title = "Title"):
    """
    Plots a spectrogram with the segmentation annotation and the estimated segmentation.
    """
    if factor.shape[0] == factor.shape[1]:
        plt.figure(figsize=(7,7))
    plt.title(title)
    padded_fac = pad_factor(factor)
    plt.pcolormesh(np.arange(padded_fac.shape[1]), np.arange(padded_fac.shape[0]), padded_fac, cmap=cm.Greys)
    plt.gca().invert_yaxis()
    for x in annotations:
        plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = "#8080FF")
    for x in predicted_ends:
        if x in annotations:
            plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = "green")#"#17becf")
        else:
            plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = "orange")
    plt.show()
    
def plot_segments_with_annotations(seg, annot):
    """
    Plots the estimated labelling of segments next to with the frontiers in the annotation.
    """
    for x in seg:
        plt.plot([x[0], x[1]], [x[2]/10,x[2]/10], '-', linewidth=1, color = "black")
    for x in annot:
        plt.plot([x[1], x[1]], [0,np.amax(np.array(seg)[:,2])/10], '-', linewidth=1, color = "red")
    plt.show()
    
  