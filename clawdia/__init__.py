"""CLAWDIA: Classification of Waves via Dictionary-based Algorithms

CLAWDIA is a modular pipeline for analyzing gravitational-wave (GW) data using
Sparse Dictionary Learning (SDL) techniques. It facilitates tasks such as
denoising and classification, offering a flexible framework that can be used
either as an integrated pipeline or as standalone routines. The primary goal
of CLAWDIA is to enhance GW signals by reducing noise and accurately classifying
them based on their astrophysical or instrumental origins.

The workflow is divided into two main stages:

1. **Denoising**: Reduces noise artifacts while preserving the key features of
   GW signals using dictionaries optimized for sparse reconstruction.
2. **Classification**: Categorizes the enhanced signals into specific groups
   using the Low-Rank Shared Dictionary Learning (LRSDL) model, leveraging
   patterns learned during the denoising stage.

CLAWDIA's modular design ensures adaptability to a wide range of applications,
making it a versatile tool for gravitational-wave data analysis.

Notes
-----
CLAWDIA was developed as part of the PhD thesis:
    _Gravitational-wave signal denoising, reconstruction and classification via
    sparse dictionary learning_ (2025).
Future updates will focus on enhancements to dictionary training and further 
    modularization of utility functions to improve usability and performance.

Submodules
----------
dictionaries
    Classes and functions for handling dictionary models, including SPAMS- and 
    LRSDL-based dictionaries, for Sparse Dictionary Learning.
estimators
    Tools for computing signal metrics such as mean squared error, 
    signal-to-noise ratio, and other statistical and signal-processing metrics.
lib
    Utility functions for mathematical operations, signal normalization, 
    and optimization routines.
pipeline
    A minimal implementation of CLAWDIA's classification pipeline, assuming 
    pre-trained dictionaries and configured parameters.
plotting
    Visualization tools for debugging and presenting results, such as 
    dictionary atoms and confusion matrices.

See Also
--------
numpy : Fundamental package for numerical computations.
scipy : Library for scientific computing with Python.
sklearn : Machine learning framework used in CLAWDIA.
spams : Sparse Modeling Software for dictionary learning and sparse coding.

"""
from . import dictionaries
from . import estimators
from . import lib
from . import pipeline


__version__ = '0.4.2'