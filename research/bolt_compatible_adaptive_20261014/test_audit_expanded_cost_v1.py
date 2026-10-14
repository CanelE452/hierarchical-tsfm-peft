"""The dense CPU pair uses two positions, unlike the six-route GPU trial."""
import unittest

import audit_expanded_cost_v1 as audit


class TwoPositionBalanceFixtures(unittest.TestCase):
    def test_each_of_two_positions_occurs_three_times(self):
        self.assertTrue(audit.verify_two_positions([0, 1, 0, 1, 0, 1]))

    def test_six_route_schedule_is_not_valid_for_a_two_route_trial(self):
        with self.assertRaisesRegex(ValueError, "TWO_POSITIONS"):
            audit.verify_two_positions([0, 1, 2, 3, 4, 5])

    def test_uneven_position_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "TWO_POSITIONS"):
            audit.verify_two_positions([0, 1, 0, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()
