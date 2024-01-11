"""dataset.py

Includes the Dataset class.

"""
import numpy as np
import pandas as pd
import pickle
from scipy.signal.windows import tukey
from sklearn.model_selection import train_test_split
import yaml


class Dataset:
    """Base class for building and managing the synthetic glitches dataset.

    Workflow, which can be run whole with `Dataset.setup_all(train_size)`:
        1- Generate (random sample) all metadata, i.e. the waveforms'
           parameters:
                `Dataset.gen_metadata()`
        2- Generate the actual dataset, i.e. waveforms, as time series:
                `Dataset.gen_dataset()`
        3- Split the dataset into train / test subsets:
                `Dataset.train_test_split(train_size)

    Notes:
        - Times begin at 0, adjust 'peak_time' w.r.t. this choice.
        - Assumes balanced classes.

    """
    classes = {'SG': 'Sine Gaussian', 'G': 'Gaussian', 'RD': 'Ring-Down'}
    n_classes = len(classes)

    def __init__(self, *, n_samples, length, peak_time, amp_threshold, tukey_alpha,
                 sample_rate, wave_parameters_limits, train_size, random_seed=None):
        self.n_samples = n_samples
        self.n_samples_total = self.n_samples * self.n_classes
        self.length = length
        self.sample_rate = sample_rate
        self.wave_limits = wave_parameters_limits
        self.peak_time = peak_time  # Reffered to as 't0' at waveform functions.
        self.tukey_alpha = tukey_alpha
        self.amp_threshold = amp_threshold
        self.train_size = train_size
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        self.metadata = None
        self.dataset = None
        self.times = None

        # Timeseries data:
        self.Xtrain = None
        self.Xtest = None
        # Labels:
        self.Ytrain = None
        self.Ytest = None
        # Indices w.r.t. 'self.dataset':
        self.Itrain = None
        self.Itest = None

    def sine_gaussian_waveform(self, times, *, t0, f0, Q, hrss, **_):
        """Note: Other kwargs will be ignored for compatibility reasons."""

        h0  = np.sqrt(np.sqrt(2) * np.pi * f0 / Q) * hrss
        env = h0 * np.exp( -(np.pi * f0 / Q * (times-t0)) ** 2)
        arg = 2 * np.pi * f0 * (times - t0)
        
        return env * np.sin(arg)

    def gaussian_waveform(self, times, *, t0, hrss, duration, **_):
        """Note: Other kwargs will be ignored for compatibility reasons.
        
        T: float
            Duration in seconds.

        """
        thres = self.amp_threshold
        T = duration
        h0  = (-8*np.log(thres))**(1/4) * hrss / np.sqrt(T)
        env = h0 * np.exp(4 * np.log(thres) * ((times-t0) / T)**2)

        return env

    def ring_down_waveform(self, times, *, t0, f0, Q, hrss, **_):
        """Note: Other kwargs will be ignored for compatibility reasons.

        This waveform has its peak at the beginning, therefore it needs to be
        manually displaced to be centered with the rest of the waveforms' peaks.

        """
        t0_ = 0
        h0  = np.sqrt(np.sqrt(2) * np.pi * f0 / Q) * hrss
        env = h0 * np.exp(- np.pi / np.sqrt(2) * f0 / Q * (times - t0_))
        arg = 2 * np.pi * f0 * (times - t0_)

        h = env * np.sin(arg)
        pad = int(t0 * self.sample_rate)
        h = np.pad(h, pad_width=(pad, 0))[:-pad]  # Largest wave will be trimmed
                                                  # at its end by 0.03s.

        return h

    wave_functions = {
        'SG': sine_gaussian_waveform,
        'G': gaussian_waveform,
        'RD': ring_down_waveform
    }

    def gen_metadata(self):
        md = pd.DataFrame(
            np.zeros((self.n_samples_total, 5), dtype=float),
            columns=('Class', 'f0', 'Q', 'hrss', 'duration')
        )
        for iclass, class_ in enumerate(self.classes):
            for i in range(iclass*self.n_samples, (iclass+1)*self.n_samples):
                f0, Q, hrss, duration = self._gen_parameters[class_](self)
                md.at[i, 'Class'] = class_
                md.at[i, 'f0'] = f0
                md.at[i, 'Q'] = Q
                md.at[i, 'hrss'] = hrss
                md.at[i, 'duration'] = duration  # Will be adjusted afterwards to take into account
                                                 # the amplitude threshold.

        self.metadata = md

    def _gen_parameters_sine_gaussian(self):
        """Generate random parameters for a single Sine Gaussian."""

        lims = self.wave_limits
        thres = self.amp_threshold
        f0   = self.rng.integers(lims['mf0'], lims['Mf0'])  # Central frequency
        Q    = self.rng.integers(lims['mQ'], lims['MQ']+1)  # Quality factor
        hrss = self.rng.uniform(lims['mhrss'], lims['Mhrss'])
        duration = 2 * Q / (np.pi * f0) * np.sqrt(-np.log(thres))
        
        return (f0, Q, hrss, duration)

    def _gen_parameters_gaussian(self):
        """Generate random parameters for a single Gaussian."""

        lims = self.wave_limits
        f0   = None  #  Casted to np.nan afterwards.
        Q    = None  #-/
        hrss = self.rng.uniform(lims['mhrss'], lims['Mhrss'])
        duration = self.rng.uniform(lims['mT'], lims['MT'])  # Duration
        
        return (f0, Q, hrss, duration)

    def _gen_parameters_ring_down(self):
        """Generate random parameters for a single Ring-Down."""

        lims = self.wave_limits
        thres = self.amp_threshold
        f0   = self.rng.integers(lims['mf0'], lims['Mf0'])  # Central frequency
        Q    = self.rng.integers(lims['mQ'], lims['MQ']+1)  # Quality factor
        hrss = self.rng.uniform(lims['mhrss'], lims['Mhrss'])
        duration = -np.sqrt(2) * Q / (np.pi * f0) * np.log(thres)
        
        return (f0, Q, hrss, duration)

    _gen_parameters = {
        'SG': _gen_parameters_sine_gaussian,
        'G': _gen_parameters_gaussian,
        'RD': _gen_parameters_ring_down
    }

    def gen_dataset(self):
        if self.metadata is None:
            raise AttributeError("'metadata' needs to be generated first!")

        self.times = np.arange(0, self.length/self.sample_rate, 1/self.sample_rate, dtype=np.float32)
        self.dataset = np.empty((self.n_samples_total, self.length), dtype=np.float32)
        
        for i in range(self.n_samples_total):
            params = self.metadata.loc[i].to_dict()
            class_ = params['Class']
            self.dataset[i] = self.wave_functions[class_](
                self, self.times, t0=self.peak_time, **params
            )

        self._apply_threshold_windowing()

    def _apply_threshold_windowing(self):
        """Shrink waves in the dataset and update its duration in the metadata.

        Shrink them according to their pre-computed duration in the metadata to
        avoid almost-but-not-zero edges, and correct those marginal durations
        longer than the window.

        """
        for i in range(self.n_samples_total):
            duration = self.metadata.at[i,'duration']
            ref_length = int(duration * self.sample_rate)
            
            if self.metadata.at[i,'Class'] == 'RD':
                # Ring-Down waves begin at the center.
                i0 = self.length // 2
                i1 = i0 + ref_length
            else:
                # SG and G are both centered.
                i0 = (self.length - ref_length) // 2
                i1 = self.length - i0

            new_lenght = i1 - i0
            if i0 < 0:
                new_lenght += i0
                i0 = 0
            if i1 > self.length:
                new_lenght -= i1 - self.length
                i1 = self.length

            window = tukey(new_lenght, alpha=self.tukey_alpha)
            self.dataset[i,:i0] = 0
            self.dataset[i,i0:i1] *= window
            self.dataset[i,i1:] = 0

            self.metadata.at[i,'duration'] = new_lenght / self.sample_rate



    def train_test_split(self):
        labels = np.repeat(range(self.n_classes), self.n_samples)
        indices = range(self.n_samples_total)  # keep track of samples after shuffle.
        
        Xtrain, Xtest, Ytrain, Ytest, Itrain, Itest = train_test_split(
            self.dataset, labels, indices, train_size=self.train_size,
            random_state=self.random_seed, shuffle=True, stratify=labels
        )

        # Timeseries data:
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        # Labels:
        self.Ytrain = Ytrain
        self.Ytest = Ytest
        # Indices:
        self.Itrain = Itrain
        self.Itest = Itest

    def setup_all(self):
        self.gen_metadata()
        self.gen_dataset()
        self.train_test_split()

