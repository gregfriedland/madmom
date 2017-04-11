# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains downbeat and bar tracking related functionality.

"""

from __future__ import absolute_import, division, print_function

import sys
import warnings

import numpy as np

from .beats_hmm import (BarStateSpace, BarTransitionModel,
                        GMMPatternTrackingObservationModel,
                        MultiPatternStateSpace,
                        MultiPatternTransitionModel,
                        RNNBeatTrackingObservationModel,
                        RNNDownBeatTrackingObservationModel, )
from ..ml.hmm import HiddenMarkovModel
from ..processors import ParallelProcessor, Processor, SequentialProcessor


def filter_downbeats(beats):
    """

    Parameters
    ----------
    beats : numpy array, shape (num_beats, 2)
        Array with beats and their position inside the bar as the second
        column.

    Returns
    -------
    downbeats : numpy array
        Array with downbeat times.

    """
    # return only downbeats (timestamps)
    return beats[beats[:, 1] == 1][:, 0]


# downbeat tracking, i.e. track beats and downbeats directly from signal
class RNNDownBeatProcessor(SequentialProcessor):
    """
    Processor to get a joint beat and downbeat activation function from
    multiple RNNs.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "Joint Beat and Downbeat Tracking with Recurrent Neural Networks"
           Proceedings of the 17th International Society for Music Information
           Retrieval Conference (ISMIR), 2016.

    Examples
    --------
    Create a RNNDownBeatProcessor and pass a file through the processor.
    The returned 2d array represents the probabilities at each frame, sampled
    at 100 frames per second. The columns represent 'beat' and 'downbeat'.

    >>> proc = RNNDownBeatProcessor()
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.beats.RNNDownBeatProcessor object at 0x...>
    >>> proc('tests/data/audio/sample.wav')
    ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[ 0.00011,  0.00037],
           [ 0.00008,  0.00043],
           ...,
           [ 0.00791,  0.00169],
           [ 0.03425,  0.00494]], dtype=float32)

    """

    def __init__(self, **kwargs):
        # pylint: disable=unused-argument
        from functools import partial
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.spectrogram import (
            FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
            SpectrogramDifferenceProcessor)
        from ..ml.nn import NeuralNetworkEnsemble
        from ..models import DOWNBEATS_BLSTM

        # define pre-processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        # process the multi-resolution spec & diff in parallel
        multi = ParallelProcessor([])
        frame_sizes = [1024, 2048, 4096]
        num_bands = [3, 6, 12]
        for frame_size, num_bands in zip(frame_sizes, num_bands):
            frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
            stft = ShortTimeFourierTransformProcessor()  # caching FFT window
            filt = FilteredSpectrogramProcessor(
                num_bands=num_bands, fmin=30, fmax=17000, norm_filters=True)
            spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
            diff = SpectrogramDifferenceProcessor(
                diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
            # process each frame size with spec and diff sequentially
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
        # stack the features and processes everything sequentially
        pre_processor = SequentialProcessor((sig, multi, np.hstack))
        # process the pre-processed signal with a NN ensemble
        nn = NeuralNetworkEnsemble.load(DOWNBEATS_BLSTM, **kwargs)
        # use only the beat & downbeat (i.e. remove non-beat) activations
        act = partial(np.delete, obj=0, axis=1)
        # instantiate a SequentialProcessor
        super(RNNDownBeatProcessor, self).__init__((pre_processor, nn, act))


def _process_dbn(process_tuple):
    """
    Extract the best path through the state space in an observation sequence.

    This proxy function is necessary to process different sequences in parallel
    using the multiprocessing module.

    Parameters
    ----------
    process_tuple : tuple
        Tuple with (HMM, observations).

    Returns
    -------
    path : numpy array
        Best path through the state space.
    log_prob : float
        Log probability of the path.

    """
    # pylint: disable=no-name-in-module
    return process_tuple[0].viterbi(process_tuple[1])


class DBNDownBeatTrackingProcessor(Processor):
    """
    Downbeat tracking with RNNs and a dynamic Bayesian network (DBN)
    approximated by a Hidden Markov Model (HMM).

    Parameters
    ----------
    beats_per_bar : int or list
        Number of beats per bar to be modeled. Can be either a single number
        or a list or array with bar lengths (in beats).
    min_bpm : float or list, optional
        Minimum tempo used for beat tracking [bpm]. If a list is given, each
        item corresponds to the number of beats per bar at the same position.
    max_bpm : float or list, optional
        Maximum tempo used for beat tracking [bpm]. If a list is given, each
        item corresponds to the number of beats per bar at the same position.
    num_tempi : int or list, optional
        Number of tempi to model; if set, limit the number of tempi and use a
        log spacing, otherwise a linear spacing. If a list is given, each
        item corresponds to the number of beats per bar at the same position.
    transition_lambda : float or list, optional
        Lambda for the exponential tempo change distribution (higher values
        prefer a constant tempo from one beat to the next one).  If a list is
        given, each item corresponds to the number of beats per bar at the
        same position.
    observation_lambda : int, optional
        Split one (down-)beat period into `observation_lambda` parts, the first
        representing (down-)beat states and the remaining non-beat states.
    threshold : float, optional
        Threshold the RNN (down-)beat activations before Viterbi decoding.
    correct : bool, optional
        Correct the beats (i.e. align them to the nearest peak of the
        (down-)beat activation function).
    fps : float, optional
        Frames per second.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "Joint Beat and Downbeat Tracking with Recurrent Neural Networks"
           Proceedings of the 17th International Society for Music Information
           Retrieval Conference (ISMIR), 2016.

    Examples
    --------
    Create a DBNDownBeatTrackingProcessor. The returned array represents the
    positions of the beats and their position inside the bar. The position is
    given in seconds, thus the expected sampling rate is needed. The position
    inside the bar follows the natural counting and starts at 1.

    The number of beats per bar which should be modelled must be given, all
    other parameters (e.g. tempo range) are optional but must have the same
    length as `beats_per_bar`, i.e. must be given for each bar length.

    >>> proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.beats.DBNDownBeatTrackingProcessor object at 0x...>

    Call this DBNDownBeatTrackingProcessor with the beat activation function
    returned by RNNDownBeatProcessor to obtain the beat positions.

    >>> act = RNNDownBeatProcessor()('tests/data/audio/sample.wav')
    >>> proc(act)  # doctest: +ELLIPSIS
    array([[ 0.09,  1.  ],
           [ 0.45,  2.  ],
           ...,
           [ 2.14,  3.  ],
           [ 2.49,  4.  ]])

    """

    MIN_BPM = 55.
    MAX_BPM = 215.
    NUM_TEMPI = 60
    TRANSITION_LAMBDA = 100
    OBSERVATION_LAMBDA = 16
    THRESHOLD = 0.05
    CORRECT = True

    def __init__(self, beats_per_bar, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 num_tempi=NUM_TEMPI, transition_lambda=TRANSITION_LAMBDA,
                 observation_lambda=OBSERVATION_LAMBDA, threshold=THRESHOLD,
                 correct=CORRECT, fps=None, **kwargs):
        # pylint: disable=unused-argument
        # pylint: disable=no-name-in-module
        # expand arguments to arrays
        beats_per_bar = np.array(beats_per_bar, ndmin=1)
        min_bpm = np.array(min_bpm, ndmin=1)
        max_bpm = np.array(max_bpm, ndmin=1)
        num_tempi = np.array(num_tempi, ndmin=1)
        transition_lambda = np.array(transition_lambda, ndmin=1)
        # make sure the other arguments are long enough by repeating them
        # TODO: check if they are of length 1?
        if len(min_bpm) != len(beats_per_bar):
            min_bpm = np.repeat(min_bpm, len(beats_per_bar))
        if len(max_bpm) != len(beats_per_bar):
            max_bpm = np.repeat(max_bpm, len(beats_per_bar))
        if len(num_tempi) != len(beats_per_bar):
            num_tempi = np.repeat(num_tempi, len(beats_per_bar))
        if len(transition_lambda) != len(beats_per_bar):
            transition_lambda = np.repeat(transition_lambda,
                                          len(beats_per_bar))
        if not (len(min_bpm) == len(max_bpm) == len(num_tempi) ==
                len(beats_per_bar) == len(transition_lambda)):
            raise ValueError('`min_bpm`, `max_bpm`, `num_tempi`, `num_beats` '
                             'and `transition_lambda` must all have the same '
                             'length.')
        # get num_threads from kwargs
        num_threads = min(len(beats_per_bar), kwargs.get('num_threads', 1))
        # init a pool of workers (if needed)
        self.map = map
        if num_threads != 1:
            import multiprocessing as mp
            self.map = mp.Pool(num_threads).map
        # convert timing information to construct a beat state space
        min_interval = 60. * fps / max_bpm
        max_interval = 60. * fps / min_bpm
        # model the different bar lengths
        self.hmms = []
        for b, beats in enumerate(beats_per_bar):
            st = BarStateSpace(beats, min_interval[b], max_interval[b],
                               num_tempi[b])
            tm = BarTransitionModel(st, transition_lambda[b])
            om = RNNDownBeatTrackingObservationModel(st, observation_lambda)
            self.hmms.append(HiddenMarkovModel(tm, om))
        # save variables
        self.beats_per_bar = beats_per_bar
        self.threshold = threshold
        self.correct = correct
        self.fps = fps

    def process(self, activations, **kwargs):
        """
        Detect the beats in the given activation function.

        Parameters
        ----------
        activations : numpy array
            (down-)beat activation function.

        Returns
        -------
        beats : numpy array
            Detected (down-)beat positions [seconds] and beat numbers.

        """
        # pylint: disable=arguments-differ
        import itertools as it
        # use only the activations > threshold (init offset to be added later)
        first = 0
        if self.threshold:
            idx = np.nonzero(activations >= self.threshold)[0]
            if idx.any():
                first = max(first, np.min(idx))
                last = min(len(activations), np.max(idx) + 1)
            else:
                last = first
            activations = activations[first:last]
        # return no beats if no activations given / remain after thresholding
        if not activations.any():
            return np.empty((0, 2))
        # (parallel) decoding of the activations with HMM
        results = list(self.map(_process_dbn, zip(self.hmms,
                                                  it.repeat(activations))))
        # choose the best HMM (highest log probability)
        best = np.argmax(np.asarray(results)[:, 1])
        # the best path through the state space
        path, _ = results[best]
        # the state space and observation model of the best HMM
        st = self.hmms[best].transition_model.state_space
        om = self.hmms[best].observation_model
        # the positions inside the pattern (0..num_beats)
        positions = st.state_positions[path]
        # corresponding beats (add 1 for natural counting)
        beat_numbers = positions.astype(int) + 1
        if self.correct:
            beats = np.empty(0, dtype=np.int)
            # for each detection determine the "beat range", i.e. states where
            # the pointers of the observation model are >= 1
            beat_range = om.pointers[path] >= 1
            # get all change points between True and False (cast to int before)
            idx = np.nonzero(np.diff(beat_range.astype(np.int)))[0] + 1
            # if the first frame is in the beat range, add a change at frame 0
            if beat_range[0]:
                idx = np.r_[0, idx]
            # if the last frame is in the beat range, append the length of the
            # array
            if beat_range[-1]:
                idx = np.r_[idx, beat_range.size]
            # iterate over all regions
            if idx.any():
                for left, right in idx.reshape((-1, 2)):
                    # pick the frame with the highest activations value
                    # Note: we look for both beats and down-beat activations;
                    #       since np.argmax works on the flattened array, we
                    #       need to divide by 2
                    peak = np.argmax(activations[left:right]) // 2 + left
                    beats = np.hstack((beats, peak))
        else:
            # transitions are the points where the beat numbers change
            # FIXME: we might miss the first or last beat!
            #        we could calculate the interval towards the beginning/end
            #        to decide whether to include these points
            beats = np.nonzero(np.diff(beat_numbers))[0] + 1
        # return the beat positions (converted to seconds) and beat numbers
        return np.vstack(((beats + first) / float(self.fps),
                          beat_numbers[beats])).T

    @staticmethod
    def add_arguments(parser, beats_per_bar, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                      num_tempi=NUM_TEMPI, transition_lambda=TRANSITION_LAMBDA,
                      observation_lambda=OBSERVATION_LAMBDA,
                      threshold=THRESHOLD, correct=CORRECT):
        """
        Add DBN downbeat tracking related arguments to an existing parser
        object.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        beats_per_bar : int or list, optional
            Number of beats per bar to be modeled. Can be either a single
            number or a list with bar lengths (in beats).
        min_bpm : float or list, optional
            Minimum tempo used for beat tracking [bpm]. If a list is given,
            each item corresponds to the number of beats per bar at the same
            position.
        max_bpm : float or list, optional
            Maximum tempo used for beat tracking [bpm]. If a list is given,
            each item corresponds to the number of beats per bar at the same
            position.
        num_tempi : int or list, optional
            Number of tempi to model; if set, limit the number of tempi and use
            a log spacing, otherwise a linear spacing. If a list is given,
            each item corresponds to the number of beats per bar at the same
            position.
        transition_lambda : float or list, optional
            Lambda for the exponential tempo change distribution (higher values
            prefer a constant tempo over a tempo change from one beat to the
            next one). If a list is given, each item corresponds to the number
            of beats per bar at the same position.
        observation_lambda : float, optional
            Split one (down-)beat period into `observation_lambda` parts, the
            first representing (down-)beat states and the remaining non-beat
            states.
        threshold : float, optional
            Threshold the RNN (down-)beat activations before Viterbi decoding.
        correct : bool, optional
            Correct the beats (i.e. align them to the nearest peak of the
            (down-)beat activation function).

        Returns
        -------
        parser_group : argparse argument group
            DBN downbeat tracking argument parser group

        """
        # pylint: disable=arguments-differ
        from ..utils import OverrideDefaultListAction

        # add DBN parser group
        g = parser.add_argument_group('dynamic Bayesian Network arguments')
        # add a transition parameters
        g.add_argument('--beats_per_bar', action=OverrideDefaultListAction,
                       default=beats_per_bar, type=int, sep=',',
                       help='number of beats per bar to be modeled (comma '
                            'separated list of bar length in beats) '
                            '[default=%(default)s]')
        g.add_argument('--min_bpm', action=OverrideDefaultListAction,
                       default=min_bpm, type=float, sep=',',
                       help='minimum tempo (comma separated list with one '
                            'value per bar length) [bpm, default=%(default)s]')
        g.add_argument('--max_bpm', action=OverrideDefaultListAction,
                       default=max_bpm, type=float, sep=',',
                       help='maximum tempo (comma separated list with one '
                            'value per bar length) [bpm, default=%(default)s]')
        g.add_argument('--num_tempi', action=OverrideDefaultListAction,
                       default=num_tempi, type=int, sep=',',
                       help='limit the number of tempi; if set, align the '
                            'tempi with log spacings, otherwise linearly '
                            '(comma separated list with one value per bar '
                            'length) [default=%(default)s]')
        g.add_argument('--transition_lambda',
                       action=OverrideDefaultListAction,
                       default=transition_lambda, type=float, sep=',',
                       help='lambda of the tempo transition distribution; '
                            'higher values prefer a constant tempo over a '
                            'tempo change from one beat to the next one ('
                            'comma separated list with one value per bar '
                            'length) [default=%(default)s]')
        # observation model stuff
        g.add_argument('--observation_lambda', action='store', type=float,
                       default=observation_lambda,
                       help='split one (down-)beat period into N parts, the '
                            'first representing beat states and the remaining '
                            'non-beat states [default=%(default)i]')
        g.add_argument('-t', dest='threshold', action='store', type=float,
                       default=threshold,
                       help='threshold the observations before Viterbi '
                            'decoding [default=%(default).2f]')
        # option to correct the beat positions
        if correct is True:
            g.add_argument('--no_correct', dest='correct',
                           action='store_false', default=correct,
                           help='do not correct the (down-)beat positions '
                                '(i.e. do not align them to the nearest peak '
                                'of the (down-)beat activation function)')
        elif correct is False:
            g.add_argument('--correct', dest='correct',
                           action='store_true', default=correct,
                           help='correct the (down-)beat positions (i.e. '
                                'align them to the nearest peak of the '
                                '(down-)beat  activation function)')
        # add output format stuff
        g = parser.add_argument_group('output arguments')
        g.add_argument('--downbeats', action='store_true', default=False,
                       help='output only the downbeats')
        # return the argument group so it can be modified if needed
        return g


class PatternTrackingProcessor(Processor):
    """
    Pattern tracking with a dynamic Bayesian network (DBN) approximated by a
    Hidden Markov Model (HMM).

    Parameters
    ----------
    pattern_files : list
        List of files with the patterns (including the fitted GMMs and
        information about the number of beats).
    min_bpm : list, optional
        Minimum tempi used for pattern tracking [bpm].
    max_bpm : list, optional
        Maximum tempi used for pattern tracking [bpm].
    num_tempi : int or list, optional
        Number of tempi to model; if set, limit the number of tempi and use a
        log spacings, otherwise a linear spacings.
    transition_lambda : float or list, optional
        Lambdas for the exponential tempo change distributions (higher values
        prefer constant tempi from one beat to the next .one)
    fps : float, optional
        Frames per second.

    Notes
    -----
    `min_bpm`, `max_bpm`, `num_tempo_states`, and `transition_lambda` must
    contain as many items as rhythmic patterns are modeled (i.e. length of
    `pattern_files`).
    If a single value is given for `num_tempo_states` and `transition_lambda`,
    this value is used for all rhythmic patterns.

    Instead of the originally proposed state space and transition model for
    the DBN [1]_, the more efficient version proposed in [2]_ is used.

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "Rhythmic Pattern Modeling for Beat and Downbeat Tracking in Musical
           Audio",
           Proceedings of the 15th International Society for Music Information
           Retrieval Conference (ISMIR), 2013.
    .. [2] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    Examples
    --------
    Create a PatternTrackingProcessor from the given pattern files. These
    pattern files include fitted GMMs for the observation model of the HMM.
    The returned array represents the positions of the beats and their position
    inside the bar. The position is given in seconds, thus the expected
    sampling rate is needed. The position inside the bar follows the natural
    counting and starts at 1.

    >>> from madmom.models import PATTERNS_BALLROOM
    >>> proc = PatternTrackingProcessor(PATTERNS_BALLROOM, fps=50)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.beats.PatternTrackingProcessor object at 0x...>

    Call this PatternTrackingProcessor with a multi-band spectrogram to obtain
    the beat and downbeat positions. The parameters of the spectrogram have to
    correspond to those used to fit the GMMs.

    >>> from madmom.processors import SequentialProcessor
    >>> from madmom.audio.spectrogram import LogarithmicSpectrogramProcessor, \
SpectrogramDifferenceProcessor, MultiBandSpectrogramProcessor
    >>> log = LogarithmicSpectrogramProcessor()
    >>> diff = SpectrogramDifferenceProcessor(positive_diffs=True)
    >>> mb = MultiBandSpectrogramProcessor(crossover_frequencies=[270])
    >>> pre_proc = SequentialProcessor([log, diff, mb])

    >>> act = pre_proc('tests/data/audio/sample.wav')
    >>> proc(act)  # doctest: +ELLIPSIS
    array([[ 0.82,  4.  ],
           [ 1.78,  1.  ],
           ...,
           [ 3.7 ,  3.  ],
           [ 4.66,  4.  ]])
    """
    # TODO: this should not be lists (lists are mutable!)
    MIN_BPM = (55, 60)
    MAX_BPM = (205, 225)
    NUM_TEMPI = None
    # Note: if multiple values are given, the individual values represent the
    #       lambdas for each transition into the beat at this index position
    TRANSITION_LAMBDA = 100

    def __init__(self, pattern_files, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 num_tempi=NUM_TEMPI, transition_lambda=TRANSITION_LAMBDA,
                 fps=None, **kwargs):
        # pylint: disable=unused-argument
        # pylint: disable=no-name-in-module
        import pickle
        min_bpm = np.array(min_bpm, ndmin=1)
        max_bpm = np.array(max_bpm, ndmin=1)
        num_tempi = np.array(num_tempi, ndmin=1)
        transition_lambda = np.array(transition_lambda, ndmin=1)
        # make sure arguments are given for each pattern (expand if needed)
        if len(min_bpm) != len(pattern_files):
            min_bpm = np.repeat(min_bpm, len(pattern_files))
        if len(max_bpm) != len(pattern_files):
            max_bpm = np.repeat(max_bpm, len(pattern_files))
        if len(num_tempi) != len(pattern_files):
            num_tempi = np.repeat(num_tempi, len(pattern_files))
        if len(transition_lambda) != len(pattern_files):
            transition_lambda = np.repeat(transition_lambda,
                                          len(pattern_files))
        # check if all lists have the same length
        if not (len(min_bpm) == len(max_bpm) == len(num_tempi) ==
                len(transition_lambda) == len(pattern_files)):
            raise ValueError('`min_bpm`, `max_bpm`, `num_tempi` and '
                             '`transition_lambda` must have the same length '
                             'as number of patterns.')
        # save some variables
        self.fps = fps
        self.num_beats = []
        # convert timing information to construct a state space
        min_interval = 60. * self.fps / np.asarray(max_bpm)
        max_interval = 60. * self.fps / np.asarray(min_bpm)
        # collect beat/bar state spaces, transition models, and GMMs
        state_spaces = []
        transition_models = []
        gmms = []
        # check that at least one pattern is given
        if not pattern_files:
            raise ValueError('at least one rhythmical pattern must be given.')
        # load the patterns
        for p, pattern_file in enumerate(pattern_files):
            with open(pattern_file, 'rb') as f:
                # Python 2 and 3 behave differently
                try:
                    # Python 3
                    pattern = pickle.load(f, encoding='latin1')
                except TypeError:
                    # Python 2 doesn't have/need the encoding
                    pattern = pickle.load(f)
            # get the fitted GMMs and number of beats
            gmms.append(pattern['gmms'])
            num_beats = pattern['num_beats']
            self.num_beats.append(num_beats)
            # model each rhythmic pattern as a bar
            state_space = BarStateSpace(num_beats, min_interval[p],
                                        max_interval[p], num_tempi[p])
            transition_model = BarTransitionModel(state_space,
                                                  transition_lambda[p])
            state_spaces.append(state_space)
            transition_models.append(transition_model)
        # create multi pattern state space, transition and observation model
        self.st = MultiPatternStateSpace(state_spaces)
        self.tm = MultiPatternTransitionModel(transition_models)
        self.om = GMMPatternTrackingObservationModel(gmms, self.st)
        # instantiate a HMM
        self.hmm = HiddenMarkovModel(self.tm, self.om, None)

    def process(self, activations, **kwargs):
        """
        Detect the beats based on the given activations.

        Parameters
        ----------
        activations : numpy array
            Activations (i.e. multi-band spectral features).

        Returns
        -------
        beats : numpy array
            Detected beat positions [seconds].

        """
        # pylint: disable=arguments-differ
        # get the best state path by calling the viterbi algorithm
        path, _ = self.hmm.viterbi(activations)
        # the positions inside the pattern (0..num_beats)
        positions = self.st.state_positions[path]
        # corresponding beats (add 1 for natural counting)
        beat_numbers = positions.astype(int) + 1
        # transitions are the points where the beat numbers change
        # FIXME: we might miss the first or last beat!
        #        we could calculate the interval towards the beginning/end to
        #        decide whether to include these points
        beat_positions = np.nonzero(np.diff(beat_numbers))[0] + 1
        # return the beat positions (converted to seconds) and beat numbers
        return np.vstack((beat_positions / float(self.fps),
                          beat_numbers[beat_positions])).T

    @staticmethod
    def add_arguments(parser, pattern_files=None, min_bpm=MIN_BPM,
                      max_bpm=MAX_BPM, num_tempi=NUM_TEMPI,
                      transition_lambda=TRANSITION_LAMBDA):
        """
        Add DBN related arguments for pattern tracking to an existing parser
        object.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        pattern_files : list
            Load the patterns from these files.
        min_bpm : list, optional
            Minimum tempi used for beat tracking [bpm].
        max_bpm : list, optional
            Maximum tempi used for beat tracking [bpm].
        num_tempi : int or list, optional
            Number of tempi to model; if set, limit the number of states and
            use log spacings, otherwise a linear spacings.
        transition_lambda : float or list, optional
            Lambdas for the exponential tempo change distribution (higher
            values prefer constant tempi from one beat to the next one).

        Returns
        -------
        parser_group : argparse argument group
            Pattern tracking argument parser group

        Notes
        -----
        `pattern_files`, `min_bpm`, `max_bpm`, `num_tempi`, and
        `transition_lambda` must the same number of items.

        """
        from ..utils import OverrideDefaultListAction
        # add GMM options
        if pattern_files is not None:
            g = parser.add_argument_group('GMM arguments')
            g.add_argument('--pattern_files', action=OverrideDefaultListAction,
                           default=pattern_files,
                           help='load the patterns (with the fitted GMMs) '
                                'from these files (comma separated list)')
        # add HMM parser group
        g = parser.add_argument_group('dynamic Bayesian Network arguments')
        g.add_argument('--min_bpm', action=OverrideDefaultListAction,
                       default=min_bpm, type=float, sep=',',
                       help='minimum tempo (comma separated list with one '
                            'value per pattern) [bpm, default=%(default)s]')
        g.add_argument('--max_bpm', action=OverrideDefaultListAction,
                       default=max_bpm, type=float, sep=',',
                       help='maximum tempo (comma separated list with one '
                            'value per pattern) [bpm, default=%(default)s]')
        g.add_argument('--num_tempi', action=OverrideDefaultListAction,
                       default=num_tempi, type=int, sep=',',
                       help='limit the number of tempi; if set, align the '
                            'tempi with log spacings, otherwise linearly '
                            '(comma separated list with one value per pattern)'
                            ' [default=%(default)s]')
        g.add_argument('--transition_lambda', action=OverrideDefaultListAction,
                       default=transition_lambda, type=float, sep=',',
                       help='lambda of the tempo transition distribution; '
                            'higher values prefer a constant tempo over a '
                            'tempo change from one bar to the next one (comma '
                            'separated list with one value per pattern) '
                            '[default=%(default)s]')
        # add output format stuff
        g = parser.add_argument_group('output arguments')
        g.add_argument('--downbeats', action='store_true', default=False,
                       help='output only the downbeats')
        # return the argument group so it can be modified if needed
        return g


# bar tracking, i.e. track downbeats from signal given beat positions
class LoadBeatsProcessor(Processor):
    """
    Load beat times from file or handle.

    """
    def __init__(self, beats, files=None, beats_suffix=None, online=False,
                 **kwargs):
        # pylint: disable=unused-argument
        from ..utils import search_files
        if online:
            self.mode = 'online'
        elif isinstance(files, list) and beats_suffix is not None:
            # overwrite beats with the files matching the suffix
            beats = search_files(files, suffix=beats_suffix)
            self.mode = 'batch'
        else:
            self.mode = 'single'
        self.beats = beats
        self.beats_suffix = beats_suffix

    def process(self, data=None, **kwargs):
        """
        Load the beats from file (handle) or read them from STDIN.

        """
        # pylint: disable=unused-argument
        if self.mode == 'online':
            return self.process_online()
        elif self.mode == 'single':
            return self.process_single()
        elif self.mode == 'batch':
            return self.process_batch(data)
        else:
            raise ValueError("don't know how to obtain the beats")

    def process_online(self):
        """
        Read the beats on a frame-by-frame basis.

        Returns
        -------
        beat : float or None
            Beat position [seconds] or None if no beat is present.

        Notes
        -----
        To be able to parse incoming beats from STDIN at the correct frame
        rate, the sender must output empty values (i.e. newlines) at the same
        rate, because this method blocks until a new value can be read in.

        """
        # pylint: disable=unused-argument
        try:
            data = float(self.beats.readline())
        except ValueError:
            data = np.empty(0)
        return data

    def process_single(self):
        """
        Load the beats in bulk-mode (i.e. all at once) from the input stream
        or file.

        Returns
        -------
        beats : numpy array
            Beat positions [seconds].

        """
        # pylint: disable=unused-argument
        from ..utils import load_events
        return load_events(self.beats)

    def process_batch(self, filename):
        """
        Load beat times from file.

        First match the given input filename to the beat filenames, then load
        the beats.

        Parameters
        ----------
        filename : str
            Input file name.

        Returns
        -------
        beats : numpy array
            Beat positions [seconds].

        Notes
        -----
        Both the file names to search for the beats as well as the suffix to
        determine the beat files must be given at instantiation time.

        """
        import os
        from ..utils import match_file

        if not isinstance(filename, str):
            raise SystemExit('Please supply a filename, not %s.' % filename)
        # select the matching beat file to a given input file from all files
        basename, ext = os.path.splitext(os.path.basename(filename))
        matches = match_file(basename, self.beats, suffix=ext,
                             match_suffix=self.beats_suffix)
        if not matches:
            raise SystemExit("can't find a beat file for %s" % filename)
        # load the beats and return them
        # TODO: Use load_beats function
        beats = np.loadtxt(matches[0])
        if beats.ndim == 2:
            # only use beat times, omit the beat positions inside the bar
            beats = beats[:, 0]
        return beats

    @staticmethod
    def add_arguments(parser, beats=sys.stdin, beats_suffix='.beats.txt'):
        """
        Add beat loading related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        beats : FileType, optional
            Where to read the beats from ('online'/'single' mode).
        beats_suffix : str, optional
            Suffix of beat files ('batch' mode)

        Returns
        -------
        argparse argument group
            Beat loading argument parser group.

        """
        import argparse
        # add beat loading options to the existing parser
        g = parser.add_argument_group('beat loading arguments')
        g.add_argument('--beats', type=argparse.FileType('rb'), default=beats,
                       help='where/how to read the beat positions from '
                            '[defaults: online: STDIN, single: STDIN]')
        g.add_argument('--beats_suffix', type=str, default=beats_suffix,
                       help='file suffix of the beat files [default: '
                            '%(default)s]')
        # return the argument group so it can be modified if needed
        return g


class SyncronizeFeaturesProcessor(Processor):
    """
    Synchronize features to beats.

    """
    def __init__(self, beat_subdivisions, feat_dim=1, fps=100, online=False,
                 **kwargs):
        # pylint: disable=unused-argument
        self.beat_subdivisions = beat_subdivisions
        # FIXME: feat_dim must be determined automatically
        self.feat_dim = feat_dim
        self.fps = fps
        self.online = online
        # sum up the feature values of one beat division
        # FIXME: does not work with arrays, only for scalars
        self.feat_sum = 0.
        # count the frames since last beat division
        self.counter = 0
        self.current_div = 0
        # length of one beat division in audio frames (depends on tempo)
        self.div_frames = None
        # store last beat time to compute tempo
        self.last_beat = None
        self.beat_features = np.zeros((beat_subdivisions, feat_dim))
        # offset the beat subdivision borders
        self.offset = 2

    def process(self, data, **kwargs):
        """
        Syncronize features to the beats.

        Parameters
        ----------
        data : tuple (beat_time(s), feature(s))

        Returns
        -------
        beat_times : None or float, or numpy array
            Beat time.
        beat_features : numpy array, shape (1, beat_subdivisions, feat_dim)
            Beat synchronous features.

        Notes
        -----
        Depending on online/offline mode the beats and features are either
        syncronized on a frame-by-frame basis or for the whole sequence,
        respectively.

        """
        if self.online:
            return self.process_online(data, **kwargs)
        return self.process_offline(data, **kwargs)

    def process_online(self, data):
        """
        This function organises the features of a piece according to the
        given beats. First, a beat interval is divided into <beat_subdivision>
        divisions. Then all features that fall into one subdivision are
        averaged. If no feature value for a subdivision is found, it is
        interpolated.

        Parameters
        ----------
        data : tuple (beat_time, feature)

        Returns
        -------
        beat_time : None or float
            Beat time (or None if no beat is present).
        beat_feat : numpy array, shape (1, beat_subdivisions, feat_dim)
            Beat synchronous features (or None if no beat is present)

        """
        beat, feature = data
        is_beat = beat.any()
        beat_feat = np.empty((0, 3))

        # init before first beat:
        if self.last_beat is None:
            if is_beat:
                # store last_beat_time
                self.last_beat = beat
            # don't return anything, since beat_feat is not synchronised yet
            return np.empty(0), beat_feat

        # init before second beat:
        if self.div_frames is None:
            if is_beat:
                # set tempo (div_frames)
                beat_interval = beat - self.last_beat
                self.div_frames = np.diff(np.round(
                    np.linspace(0, beat_interval * self.fps,
                                self.beat_subdivisions + 1)))
                self.last_beat = beat
            # don't return anything, since beat_feat is not synchronised yet
            return np.empty(0), beat_feat

        # normal action, everything is initialised
        # add feature to the cumulative sum
        self.feat_sum += feature
        self.counter += 1  # starts with 1
        # check if the current frame is the end of a beat subdivision
        is_end_div = (self.counter >= self.div_frames[self.current_div])
        if is_end_div:
            # compute mean of the features in the previous subdivision
            self.beat_features[self.current_div, :] = \
                self.feat_sum / self.counter
            # proceed to the next subdivision
            self.current_div = (self.current_div + 1) % self.beat_subdivisions
            # reset cumulative sum and the frame counter
            self.feat_sum = 0.
            self.counter = 0
        if is_beat:
            beat_feat = self.beat_features[np.newaxis, :]
            # compute new beat interval (tempo)
            beat_interval = beat - self.last_beat
            # update beat subdivision lengths
            self.div_frames = np.diff(np.round(np.linspace(
                0, beat_interval * self.fps, self.beat_subdivisions + 1)))
            # If we reset the frame_counter, we also have to modify feat_sum
            if self.counter > self.offset:
                self.feat_sum = self.feat_sum * self.offset / self.counter
            else:
                self.feat_sum = self.beat_features[-1, :] * self.offset
            # Reset frame counter. If we want to collect features before the
            # actual subdivision borders, we start counting with an offset
            self.counter = self.offset
            # store last beat time
            self.last_beat = beat
            self.current_div = 0
            # remove old entries
            self.beat_features = np.zeros((self.beat_subdivisions,
                                           self.feat_dim))
        return beat, beat_feat

    def process_offline(self, data):
        """
        This function organises the features of one song according to the
        given beats. First, a beat interval is divided into <beat_subdivision>
        divisions. Then all features that fall into one subdivision are
        summarised by a <summarise> function. If no feature value for a
        subdivision is found, it is interpolated.

        Parameters
        ----------
        data : tuple (beat_times, features)
            Tuple of two numpy arrays, the first containing the beat times in
            seconds, the second the features to be synchronised to them.

        Returns
        -------
        beat_times : numpy array
            Beat times.
        beat_features : numpy array [1, beat_subdivisions, feat_dim]
            Beat synchronous features.

        """
        beats, features = data
        # no beats, return immediately
        if beats.size == 0:
            return np.array([]), np.array([])
        # beats can be 1D (only beat times) or 2D (times, position inside bar)
        if beats.ndim > 1:
            beats = beats[:, 0]
        # trim beat sequence
        while (float(len(features)) / self.fps) < beats[-1]:
            beats = beats[:-1]
            warnings.warn('Beat sequence too long compared to features.')
        # number of beats
        num_beats = len(beats)
        # feature dimension (make sure features are 2D)
        features = np.array(features, copy=False, ndmin=2)
        feat_dim = features.shape[-1]
        # init a 3D feature aggregation array
        beat_features = np.empty((num_beats - 1, self.beat_subdivisions,
                                  feat_dim))
        # start first beat 20ms before actual annotation
        # TODO: revisit these 20ms or make it an argument?
        next_beat_frame = int(max(0, np.floor((beats[0] - 0.02) * self.fps)))
        for i in range(num_beats - 1):
            # aggregate all feature values that fall into a window of
            # length = duration_beat / beat_div, centered on the beat
            # annotations or interpolated subdivisions
            duration_beat = beats[i + 1] - beats[i]
            offset = 0.5 * duration_beat / self.beat_subdivisions
            # offset should be < 50 ms
            offset = np.min([offset, 0.05])
            # first frame of current beat
            beat_start_frame = next_beat_frame
            # first frame of next beat
            next_beat_frame = int(np.floor((beats[i + 1] - offset) * self.fps))
            # set up array with time in sec of each frame center. last frame
            # is the last frame of the last gmm of the current bar
            num_beat_frames = next_beat_frame - beat_start_frame
            # we need to put each feature frame in its corresponding
            # beat subdivison. We use a hack to get num_beat_frames equally
            # distributed bins between 0 and (self.beat_subdivision - 1)
            subdiv = np.floor(np.linspace(0, self.beat_subdivisions,
                                          num_beat_frames, endpoint=False))
            features_beat = features[beat_start_frame:next_beat_frame]
            # group features by bar position
            features_grouped = [features_beat[subdiv == x] for x in
                                np.arange(0, self.beat_subdivisions)]
            # interpolate missing feature values which occur at fast tempi
            features_grouped = self.interpolate_missing(features_grouped,
                                                        feat_dim)
            beat_features[i, :, :] = np.array([np.mean(x, axis=0) for x in
                                               features_grouped])
        return beats, beat_features

    def interpolate_missing(self, features, feat_dim):
        """
        Interpolate missing beat features.

        Parameters
        ----------
        features : list
            Features, grouped by bar position.
        feat_dim : int
            Number of feature dimensions.

        Returns
        -------
        numpy array
            Features with missing features interpolated.

        """
        nan_fill = np.empty(feat_dim) * np.nan
        means = np.array([np.mean(x, axis=0) if x.any() else nan_fill
                          for x in features])
        good_rows = np.unique(np.where(np.logical_not(np.isnan(means)))[0])
        if len(good_rows) < self.beat_subdivisions:
            bad_rows = np.unique(np.where(np.isnan(means))[0])
            # initialise missing values with empty array
            for r in bad_rows:
                features[r] = np.empty((1, feat_dim))
            for d in range(feat_dim):
                means_p = np.interp(np.arange(0, self.beat_subdivisions),
                                    good_rows,
                                    means[good_rows, d])
                for r in bad_rows:
                    features[r][0, d] = means_p[r]
        return features


class RNNBarProcessor(Processor):
    """
    Processor to retrieve a downbeat activation function from beat-synchronous
    features with multiple GRU-RNNs.

    Parameters
    ----------
    beat_subdivisions : tuple, optional
        Number of beat subdivisions for the percussive and harmonic feature.

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "Downbeat Tracking Using Beat-Synchronous Features and Recurrent
            Networks",
           Proceedings of the 17th International Society for Music Information
           Retrieval Conference (ISMIR), 2016.

    """

    def __init__(self, beat_subdivisions=(4, 2), **kwargs):
        # pylint: disable=unused-argument
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.spectrogram import (
            FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
            SpectrogramDifferenceProcessor)
        from ..audio.chroma import CLPChromaProcessor
        from ..ml.nn import NeuralNetworkEnsemble
        from ..models import DOWNBEATS_BGRU
        # define percussive feature
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        frames = FramedSignalProcessor(frame_size=2048, fps=100)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        spec = FilteredSpectrogramProcessor(
            num_bands=6, fmin=30., fmax=17000., norm_filters=True)
        log_spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
        diff = SpectrogramDifferenceProcessor(
            diff_ratio=0.5, positive_diffs=True)
        self.perc_feat = SequentialProcessor(
            (sig, frames, stft, spec, log_spec, diff))
        # define harmonic feature
        self.harm_feat = CLPChromaProcessor(
            fps=100, fmin=27.5, fmax=4200., compression_factor=100,
            norm=True, threshold=0.001)
        # sync features to the beats
        # TODO: can beat_subdivisions extracted from somewhere?
        self.perc_beat_sync = SyncronizeFeaturesProcessor(beat_subdivisions[0])
        self.harm_beat_sync = SyncronizeFeaturesProcessor(beat_subdivisions[1])
        # NN ensembles to process beat-synchronous features
        self.perc_nn = NeuralNetworkEnsemble.load(DOWNBEATS_BGRU[0])
        self.harm_nn = NeuralNetworkEnsemble.load(DOWNBEATS_BGRU[1])

    def process(self, data, **kwargs):
        # split the input data
        beats, signal = data
        # process the signal
        perc = self.perc_feat(signal)
        harm = self.harm_feat(signal)
        # sync to the beats
        perc_synced = self.perc_beat_sync((beats, perc))[1]
        harm_synced = self.harm_beat_sync((beats, harm))[1]
        # process with NNs and average the predictions
        # Note: reshape the NN input to length of synced features
        perc = self.perc_nn(perc_synced.reshape((len(perc_synced), -1)))
        harm = self.harm_nn(harm_synced.reshape((len(harm_synced), -1)))
        return beats, np.mean([perc, harm], axis=0)


class DBNBarTrackingProcessor(Processor):
    """
    Bar tracking with a dynamic Bayesian network (DBN) approximated by a
    Hidden Markov Model (HMM).

    Parameters
    ----------
    beats_per_bar : int or list
        Number of beats per bar to be modeled. Can be either a single number
        or a list or array with bar lengths (in beats).
    observation_weight : int, optional
        Weight for the downbeat activations.
    meter_change_prob : float, optional
        Probability to change meter at bar boundaries.
    online : bool, optional
        Use the forward algorithm (instead of Viterbi) to decode the downbeats.

    """

    OBSERVATION_WEIGHT = 100
    METER_CHANGE_PROB = 1e-7

    def __init__(self, beats_per_bar=(3, 4),
                 observation_weight=OBSERVATION_WEIGHT,
                 meter_change_prob=METER_CHANGE_PROB, online=False, **kwargs):
        # pylint: disable=unused-argument
        # save variables
        self.beats_per_bar = beats_per_bar
        # state space & transition model for each bar length
        state_spaces = []
        transition_models = []
        for beats in self.beats_per_bar:
            # Note: tempo and transition_lambda is not relevant
            st = BarStateSpace(beats, min_interval=1, max_interval=1)
            tm = BarTransitionModel(st, transition_lambda=1)
            state_spaces.append(st)
            transition_models.append(tm)
        # Note: treat diffrent bar lengths as different patterns and use the
        #       existing MultiPatternStateSpace and MultiPatternTransitionModel
        self.st = MultiPatternStateSpace(state_spaces)
        self.tm = MultiPatternTransitionModel(
            transition_models, pattern_change_prob=meter_change_prob)
        # observation model
        self.om = RNNBeatTrackingObservationModel(self.st, observation_weight)
        # instantiate a HMM
        self.hmm = HiddenMarkovModel(self.tm, self.om, None)
        # kepp state in online mode
        self.online = online
        if self.online:
            self.beat = np.empty((0, 2))  # [beat_time, beat_number]
            self.meter = None

    def process(self, data, **kwargs):
        """
        Decode downbeats from the given beats and activation function.

        Parameters
        ----------
        data : tuple (beat times, activation function)

        Returns
        -------
        numpy array
            Decoded (down-)beat positions [seconds] and beat numbers.

        """
        # get the best state path by calling the inference algorithm
        if self.online:
            return self.process_online(data, **kwargs)
        return self.process_offline(data, **kwargs)

    def process_online(self, data, **kwargs):
        """
        Detect downbeats from the given beats and activation function with the
        forward algorithm.

        Parameters
        ----------
        data : tuple (beat times, activation function)

        Returns
        -------
        numpy array
            Decoded (down-)beat positions [seconds] and beat numbers.

        """
        # pylint: disable=unused-argument
        # split data
        beat, activation = data
        # infer beat numbers only at beat positions
        if not beat.any():
            return np.empty((0, 2))
        # computing the forward path
        fwd = self.hmm.forward(activation)
        # use simply the most probable state
        state = np.argmax(fwd)
        # get the position inside the bar
        position = self.st.state_positions[state]
        # corresponding beat numbers (add 1 for natural counting)
        beat_number = position.astype(int) + 1
        # save the current meter
        self.meter = self.beats_per_bar[self.st.state_patterns[state]]
        # save the current beat [time, number]
        self.beat = np.append(beat, beat_number)
        # return beats and their beat numbers
        return np.array(self.beat, ndmin=2)

    def process_offline(self, data, **kwargs):
        """
        Detect downbeats from the given beats and activation function with
        Viterbi decoding.

        Parameters
        ----------
        data : tuple (beat times, activation function)

        Returns
        -------
        numpy array
            Decoded (down-)beat positions [seconds] and beat numbers.

        """
        # pylint: disable=unused-argument
        # split data
        beats, activations = data
        if not beats.any():
            return np.empty((0, 2))
        # Viterbi decoding
        path, _ = self.hmm.viterbi(activations)
        # get the position inside the bar
        position = self.st.state_positions[path]
        # the beat numbers are the counters + 1 at the transition points
        beat_numbers = position.astype(int) + 1
        # add the last beat
        last_beat_number = np.mod(beat_numbers[-1], self.beats_per_bar[
            self.st.state_patterns[path[-1]]]) + 1
        beat_numbers = np.append(beat_numbers, last_beat_number)
        # return beats and their beat numbers
        return np.vstack(zip(beats, beat_numbers))

    @classmethod
    def add_arguments(cls, parser, beats_per_bar,
                      observation_weight=OBSERVATION_WEIGHT,
                      meter_change_prob=METER_CHANGE_PROB):
        """
        Add DBN related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        beats_per_bar : int or list, optional
            Number of beats per bar to be modeled. Can be either a single
            number or a list with bar lengths (in beats).
        observation_weight : float, optional
            Weight for the activations at downbeat times.
        meter_change_prob : float, optional
            Probability to change meter at bar boundaries.

        Returns
        -------
        parser_group : argparse argument group
            DBN bar tracking argument parser group

        """
        # pylint: disable=arguments-differ
        from ..utils import OverrideDefaultListAction
        # add DBN parser group
        g = parser.add_argument_group('dynamic Bayesian Network arguments')
        g.add_argument('--beats_per_bar', action=OverrideDefaultListAction,
                       default=beats_per_bar, type=int, sep=',',
                       help='number of beats per bar to be modeled (comma '
                            'separated list of bar length in beats) '
                            '[default=%(default)s]')
        g.add_argument('--observation_weight', action='store', type=float,
                       default=observation_weight,
                       help='weight for the downbeat activations '
                            '[default=%(default)i]')
        g.add_argument('--meter_change_prob', action='store', type=float,
                       default=meter_change_prob,
                       help='meter change probability [default=%(default).g]')
        # add output format stuff
        parser = parser.add_argument_group('output arguments')
        parser.add_argument('--downbeats', action='store_true', default=False,
                            help='output only the downbeats')
        # return the argument group so it can be modified if needed
        return parser
