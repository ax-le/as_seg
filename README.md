# as_seg: module for computing and segmenting autosimilarity matrices. #

Hello, and welcome on this repository!

This project aims at computing autosimilarity matrices, and segmenting them, which consists of the task of structural segmentation.

The current version contains the CBM algorithm [1], along with a (low-effort) implementation of Foote's novelty algorithm [2].

It can be installed with pip usin `pip install as-seg`.

This is a first release, and may contain bug. Comments are welcomed!

## Tutorial notebook ##

A tutorial notebook presenting the most important components of this toolbox is available in the folder "Notebooks" (called "Walkthrough").

## Software version ##

This code was developed with Python 3.8.5, and some external libraries detailed in dependencies.txt. They should be installed automatically if this project is downloaded using pip.

## Credits ##

Code was created by Axel Marmoret (<axel.marmoret@irisa.fr>), and strongly supported by Jeremy E. Cohen (<jeremy.cohen@cnrs.fr>).

The technique in itself was also developed by Frédéric Bimbot (<bimbot@irisa.fr>).

## References ##
[1] Marmoret, A., Cohen, J. E., & Bimbot, F., "Convolutive Block-Matching Segmentation Algorithm with Application to Music Structure Analysis", 2022, arXiv preprint arXiv:2210.15356.

[2] Foote, J., "Automatic audio segmentation using a measure of audio novelty", in: 2000 IEEE Int. Conf. Multimedia and Expo. ICME2000. Proc. Latest Advances in the Fast Changing World of Multimedia, vol. 1, IEEE, 2000,pp. 452–455.
