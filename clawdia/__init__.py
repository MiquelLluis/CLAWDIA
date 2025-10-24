"""
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

CLAWDIA was developed as part of the PhD thesis:

   Llorens-Monteagudo, M., 2025, *Gravitational-wave signal denoising,
   reconstruction and classification via Sparse Dictionary Learning*,
   PhD thesis, Universitat de València, Spain.


"""
from . import dictionaries
from . import estimators
from . import lib
from . import pipeline


__version__ = '0.5.0'