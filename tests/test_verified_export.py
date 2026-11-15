import unittest
from one_gpu_lm.verified.export import inventory


class EpisodeInventory(unittest.TestCase):
    def test_action_rows_do_not_inflate_full_episode_target(self):
        episodes = [{'episode_id': 'run:a', 'task_id': 'a'}, {'episode_id': 'run:b', 'task_id': 'b'}]
        records = [{'episode_id': 'run:a'}]*60 + [{'episode_id': 'run:b'}]*60
        result = inventory(episodes, records)
        self.assertEqual(result['positive_action_records'], 120)
        self.assertEqual(result['full_episodes'], 2)
        self.assertFalse(result['minimum_100_full_episodes_met'])
        self.assertFalse(result['candidate_training_authorized_by_this_export'])

    def test_repeated_task_is_not_counted_as_a_new_task(self):
        result = inventory([{'episode_id': 'r1:a', 'task_id': 'a'}, {'episode_id': 'r2:a', 'task_id': 'a'}],
                           [{'episode_id': 'r1:a'}, {'episode_id': 'r2:a'}])
        self.assertEqual(result['full_episodes'], 2)
        self.assertEqual(result['unique_tasks'], 1)

    def test_empty_duplicate_and_orphan_records_are_rejected(self):
        episode = {'episode_id': 'r:a', 'task_id': 'a'}
        for episodes, records in [([episode], []), ([episode, episode], [{'episode_id': 'r:a'}]),
                                  ([episode], [{'episode_id': 'unregistered'}])]:
            with self.assertRaises(ValueError):
                inventory(episodes, records)


if __name__ == '__main__':
    unittest.main()
