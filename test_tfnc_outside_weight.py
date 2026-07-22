import unittest

import torch

from train_frozen_vis import _compute_tfnc_outside_weight


class TfncOutsideWeightTest(unittest.TestCase):
    def test_uniform_ablation_assigns_one_to_each_outside_frame(self):
        false_negative_prob = torch.tensor([[0.1, 0.5, 0.9]])
        outside_mask = torch.tensor([[True, False, True]])

        actual = _compute_tfnc_outside_weight(
            false_negative_prob,
            outside_mask,
            uniform_outside_weight=True,
        )

        torch.testing.assert_close(actual, torch.tensor([[1.0, 0.0, 1.0]]))

    def test_default_keeps_false_negative_attenuation(self):
        false_negative_prob = torch.tensor([[0.1, 0.5, 0.9]])
        outside_mask = torch.tensor([[True, False, True]])

        actual = _compute_tfnc_outside_weight(
            false_negative_prob,
            outside_mask,
        )

        torch.testing.assert_close(actual, torch.tensor([[0.9, 0.0, 0.1]]))


if __name__ == "__main__":
    unittest.main()
