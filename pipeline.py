import yaml

from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
import numpy as np

from . import util


def main():
    with open('run-pipeline.yaml') as f:
        cfg = yaml.safe_load(f)

    # Load data:
    # - Frame
    # - gps times + labels
    # - asd
    # - dictionaries

    # Preprocessing:
    # - Extract windows from frame
    # - Whiten + crop + norm
    # - Denoise + norm

    # Classification:
    # - Crop to fit atoms' length of the classification dictionary
    # - Predict labels

    # Write output to disk:
    # - Whitened strains
    # - Denoised strains
    # - Prediction vector
    # - F1 score
    # - Confusion matrix


if __name__ == '__main__':
    main()
    print(util.FINISHED)
