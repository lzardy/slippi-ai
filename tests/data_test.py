import pickle
from typing import Sequence
import unittest
import zlib

import numpy as np
import tree

from slippi_ai import data, embed

def reconstruct(xs: Sequence, counts: Sequence[int]) -> Sequence:
  reconstruction = []
  for x, c in zip(xs, counts):
    reconstruction.extend([x] * (c + 1))
  return reconstruction

class CompressRepeatedActionsTest(unittest.TestCase):

  def test_indices_and_counts(self):
    actions = np.random.randint(2, size=100)
    repeats = data.detect_repeated_actions(actions)
    indices, counts = data.indices_and_counts(repeats,3)

    reconstruction = []
    for i, c in zip(indices, counts):
      reconstruction.extend([actions[i]] * (c + 1))
    self.assertSequenceEqual(reconstruction, actions.tolist())

  def test_compress(self):
    rewards = np.random.uniform(size=605)

    # with open('game.json','rb') as outfile:
    #   game = pickle.load(outfile)

    with open('data/multishine/multishine.slp.pkl', 'rb')  as f:
      game = pickle.loads(zlib.decompress(f.read()))

    embedder = embed.embed_controller_discrete

    compressed1 = data.compress_repeated_actions(game,rewards,embedder,604)
    compressed2 = data.compress_repeated_actions(game,rewards,embedder,0)

    def assert_equal_with_path(path, xs1, xs2):
      rx1 = reconstruct(xs1, compressed1.counts)
      rx2 = reconstruct(xs2, compressed2.counts)
      self.assertSequenceEqual(rx1, rx2, msg='.'.join(map(str, path)))

    tree.map_structure_with_path(
      assert_equal_with_path,
      compressed1.states, compressed2.states)

if __name__ == '__main__':
  unittest.main(failfast=True)
