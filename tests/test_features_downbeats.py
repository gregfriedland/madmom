# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.features.downbeats module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os.path import join as pj

from madmom.ml.hmm import HiddenMarkovModel

from madmom.features import Activations
from madmom.audio.chroma import CLPChroma
from madmom.features.beats_hmm import *
from madmom.features.downbeats import *
from madmom.models import PATTERNS_BALLROOM
from . import AUDIO_PATH, ACTIVATIONS_PATH, ANNOTATIONS_PATH

sample_file = pj(AUDIO_PATH, "sample.wav")
sample_beats = np.loadtxt(pj(ANNOTATIONS_PATH, "sample.beats"))
sample_downbeat_act = Activations(pj(ACTIVATIONS_PATH,
                                     "sample.downbeats_blstm.npz"))


class TestRNNDownBeatProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = RNNDownBeatProcessor()

    def test_process(self):
        downbeat_act = self.processor(sample_file)
        self.assertTrue(np.allclose(downbeat_act, sample_downbeat_act,
                                    atol=1e-5))


class TestDBNDownBeatTrackingProcessorClass(unittest.TestCase):
    def setUp(self):
        self.processor = DBNDownBeatTrackingProcessor(
            [3, 4], fps=sample_downbeat_act.fps)

    def test_types(self):
        self.assertIsInstance(self.processor.correct, bool)
        # self.assertIsInstance(self.processor.st, BarStateSpace)
        # the bar lengths are modelled with individual HMMs
        self.assertIsInstance(self.processor.hmms, list)
        self.assertIsInstance(self.processor.hmms[0], HiddenMarkovModel)
        self.assertIsInstance(self.processor.hmms[0].transition_model,
                              BarTransitionModel)
        self.assertIsInstance(self.processor.hmms[0].observation_model,
                              RNNDownBeatTrackingObservationModel)

    def test_values(self):
        self.assertTrue(self.processor.correct)
        # we have to test each bar length individually
        path, prob = self.processor.hmms[0].viterbi(sample_downbeat_act)
        self.assertTrue(np.allclose(path[:13],
                                    [7682, 7683, 7684, 7685, 7686, 7687, 7688,
                                     7689, 217, 218, 219, 220, 221]))
        self.assertTrue(np.allclose(prob, -764.586595603))
        tm = self.processor.hmms[0].transition_model
        positions = tm.state_space.state_positions[path]
        self.assertTrue(np.allclose(positions[:10],
                                    [2.77142857, 2.8, 2.82857143, 2.85714286,
                                     2.88571429, 2.91428571, 2.94285714,
                                     2.97142857, 0, 0.02857143]))
        intervals = tm.state_space.state_intervals[path]
        self.assertTrue(np.allclose(intervals[:10], 35))

    def test_process(self):
        downbeats = self.processor(sample_downbeat_act)
        self.assertTrue(np.allclose(downbeats, [[0.09, 1], [0.45, 2],
                                                [0.79, 3], [1.12, 4],
                                                [1.47, 1], [1.8, 2],
                                                [2.14, 3], [2.49, 4]]))
        # set the threshold
        self.processor.threshold = 1
        downbeats = self.processor(sample_downbeat_act)
        self.assertTrue(np.allclose(downbeats, np.empty((0, 2))))


sample_pattern_features = Activations(pj(ACTIVATIONS_PATH,
                                         "sample.gmm_pattern_tracker.npz"))


class TestPatternTrackingProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = PatternTrackingProcessor(
            PATTERNS_BALLROOM, fps=sample_pattern_features.fps)

    def test_types(self):
        self.assertIsInstance(self.processor.num_beats, list)
        self.assertIsInstance(self.processor.st, MultiPatternStateSpace)
        self.assertIsInstance(self.processor.tm, MultiPatternTransitionModel)
        self.assertIsInstance(self.processor.om,
                              GMMPatternTrackingObservationModel)
        self.assertIsInstance(self.processor.hmm, HiddenMarkovModel)

    def test_values(self):
        self.assertTrue(self.processor.fps == 50)
        self.assertTrue(np.allclose(self.processor.num_beats, [3, 4]))
        path, prob = self.processor.hmm.viterbi(sample_pattern_features)
        self.assertTrue(np.allclose(path[:12], [5573, 5574, 5575, 5576, 6757,
                                                6758, 6759, 6760, 6761, 6762,
                                                6763, 6764]))
        self.assertTrue(np.allclose(prob, -468.8014))
        patterns = self.processor.st.state_patterns[path]
        self.assertTrue(np.allclose(patterns,
                                    np.ones(len(sample_pattern_features))))
        positions = self.processor.st.state_positions[path]
        self.assertTrue(np.allclose(positions[:6], [1.76470588, 1.82352944,
                                                    1.88235296, 1.94117648,
                                                    2, 2.0588236]))

    def test_process(self):
        beats = self.processor(sample_pattern_features)
        self.assertTrue(np.allclose(beats, [[0.08, 3], [0.42, 4], [0.76, 1],
                                            [1.1, 2], [1.44, 3], [1.78, 4],
                                            [2.12, 1], [2.46, 2], [2.8, 3]]))


class TestBarStateSpaceClass(unittest.TestCase):

    def test_types(self):
        bss = BarStateSpace(2, 1, 4)
        self.assertIsInstance(bss.num_beats, int)
        self.assertIsInstance(bss.num_states, int)
        # self.assertIsInstance(bss.intervals, np.ndarray)
        self.assertIsInstance(bss.state_positions, np.ndarray)
        self.assertIsInstance(bss.state_intervals, np.ndarray)
        self.assertIsInstance(bss.first_states, list)
        self.assertIsInstance(bss.last_states, list)
        # dtypes
        # self.assertTrue(bss.intervals.dtype == np.uint32)
        self.assertTrue(bss.state_positions.dtype == np.float)
        self.assertTrue(bss.state_intervals.dtype == np.uint32)

    def test_values(self):
        # 2 beats, intervals 1 to 4
        bss = BarStateSpace(2, 1, 4)
        self.assertTrue(bss.num_beats == 2)
        self.assertTrue(bss.num_states == 20)
        # self.assertTrue(np.allclose(bss.intervals, [1, 2, 3, 4]))
        self.assertTrue(np.allclose(bss.state_positions,
                                    [0, 0, 0.5, 0, 1. / 3, 2. / 3,
                                     0, 0.25, 0.5, 0.75,
                                     1, 1, 1.5, 1, 4. / 3, 5. / 3,
                                     1, 1.25, 1.5, 1.75]))
        self.assertTrue(np.allclose(bss.state_intervals,
                                    [1, 2, 2, 3, 3, 3, 4, 4, 4, 4,
                                     1, 2, 2, 3, 3, 3, 4, 4, 4, 4]))
        self.assertTrue(np.allclose(bss.first_states, [[0, 1, 3, 6],
                                                       [10, 11, 13, 16]]))
        self.assertTrue(np.allclose(bss.last_states, [[0, 2, 5, 9],
                                                      [10, 12, 15, 19]]))
        # other values: 1 beat, intervals 2 to 6
        bss = BarStateSpace(1, 2, 6)
        self.assertTrue(bss.num_beats == 1)
        self.assertTrue(bss.num_states == 20)
        # self.assertTrue(np.allclose(bss.intervals, [2, 3, 4, 5, 6]))
        self.assertTrue(np.allclose(bss.state_positions,
                                    [0, 0.5,
                                     0, 1. / 3, 2. / 3,
                                     0, 0.25, 0.5, 0.75,
                                     0, 0.2, 0.4, 0.6, 0.8,
                                     0, 1. / 6, 2. / 6, 0.5, 4. / 6, 5. / 6]))
        self.assertTrue(np.allclose(bss.state_intervals,
                                    [2, 2, 3, 3, 3, 4, 4, 4, 4,
                                     5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6]))
        self.assertTrue(np.allclose(bss.first_states, [[0, 2, 5, 9, 14]]))
        self.assertTrue(np.allclose(bss.last_states, [[1, 4, 8, 13, 19]]))


class TestBarTransitionModelClass(unittest.TestCase):

    def test_types(self):
        bss = BarStateSpace(2, 1, 4)
        tm = BarTransitionModel(bss, 100)
        self.assertIsInstance(tm, BarTransitionModel)
        self.assertIsInstance(tm, TransitionModel)
        self.assertIsInstance(tm.state_space, BarStateSpace)
        self.assertIsInstance(tm.transition_lambda, list)
        self.assertIsInstance(tm.states, np.ndarray)
        self.assertIsInstance(tm.pointers, np.ndarray)
        self.assertIsInstance(tm.probabilities, np.ndarray)
        self.assertIsInstance(tm.log_probabilities, np.ndarray)
        self.assertIsInstance(tm.num_states, int)
        self.assertIsInstance(tm.num_transitions, int)
        self.assertTrue(tm.states.dtype == np.uint32)
        self.assertTrue(tm.pointers.dtype == np.uint32)
        self.assertTrue(tm.probabilities.dtype == np.float)
        self.assertTrue(tm.log_probabilities.dtype == np.float)

    def test_values(self):
        bss = BarStateSpace(2, 1, 4)
        tm = BarTransitionModel(bss, 100)
        self.assertTrue(np.allclose(tm.states,
                                    [10, 12, 15, 1, 15, 19, 3, 4, 15, 19, 6, 7,
                                     8, 0, 2, 5, 11, 5, 9, 13, 14, 5, 9, 16,
                                     17, 18]))
        self.assertTrue(np.allclose(tm.pointers,
                                    [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14,
                                     16, 17, 19, 20, 21, 23, 24, 25, 26]))
        self.assertTrue(np.allclose(tm.probabilities,
                                    [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1,
                                     1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]))
        self.assertTrue(np.allclose(tm.log_probabilities,
                                    [0, 0, -33.33333, 0, 0, -25,
                                     0, 0, -33.33333, 0, 0, 0, 0,
                                     0, 0, -33.33333, 0, 0, -25,
                                     0, 0, -33.33333, 0, 0, 0, 0]))
        self.assertTrue(tm.num_states == 20)
        self.assertTrue(tm.num_transitions == 26)


class TestSyncronizeFeaturesProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = SyncronizeFeaturesProcessor(beat_subdivisions=2)

    def test_process(self):
        data = [sample_beats, CLPChroma(sample_file, fps=100)]
        beats, feat_sync = self.processor(data)
        target = [[0.28231065, 0.14807641, 0.22790557, 0.41458403, 0.15966462,
                   0.22294236, 0.1429988, 0.16661506, 0.5978227, 0.24039252,
                   0.23444982, 0.21910049],
                  [0.25676728, 0.13382165, 0.19957431, 0.47225753, 0.18936998,
                   0.17014103, 0.14079712, 0.18317944, 0.60692955, 0.20016842,
                   0.17619181, 0.24408179]]
        self.assertTrue(np.allclose(feat_sync[0, :], target, rtol=1e-3))


class TestRNNBarProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = RNNBarProcessor()

    def test_process(self):
        beats, act = self.processor((sample_beats, sample_file))
        self.assertTrue(np.allclose(act, [0.48194462, 0.12625194, 0.1980453],
                                    rtol=1e-3))


class TestDBNBarProcessorRNNClass(unittest.TestCase):

    def setUp(self):
        self.processor = DBNBarTrackingProcessor()
        self.rnn_outout = np.array([0.4819403, 0.1262536, 0.1980488])
        self.dbn_outout = np.array([[0.0913, 1.], [0.7997, 2.], [1.4806, 3.],
                                    [2.1478, 1.]])

    def test_dbn(self):
        # check DBN output
        path, log = self.processor.hmm.viterbi(self.rnn_outout)
        self.assertTrue(np.allclose(path, [0, 1, 2]))
        self.assertTrue(np.allclose(log, -12.2217575073))

    def test_process(self):
        beats = self.processor([sample_beats[:, 0], self.rnn_outout])
        self.assertTrue(np.allclose(beats, self.dbn_outout))
