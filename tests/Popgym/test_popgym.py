import sys
# sys.path.append('/opt/rl_transformers/environments/POPGym')
# print(sys.path)

from environments.POPGym.popgym_env import POPGymWrapper
import unittest
import numpy as np






# class TestPOPGymWrapper(unittest.TestCase):

#     def setUp(self):
#         """Создание окружения для тестов"""
#         self.env = POPGymWrapper('RepeatPreviousEasy')

#     def test_initialization(self):
#         """Проверка правильности инициализации"""
#         self.assertIsNotNone(self.env)
#         self.assertEqual(self.env.max_episode_steps, 4, "Incorrect max episode steps")
#         self.assertTrue(hasattr(self.env, 'observation_space'), "Observation space missing")
#         self.assertTrue(hasattr(self.env, 'action_space'), "Action space missing")

#     def test_reset(self):
#         """Проверка корректного сброса окружения"""
#         obs = self.env.reset()
#         self.assertIsInstance(obs, np.ndarray, "Reset should return an observation as a numpy array")
#         self.assertEqual(obs.shape, self.env.observation_space.obs_shape, "Reset observation shape mismatch")

#     def test_step(self):
#         """Проверка выполнения шага в среде"""
#         self.env.reset()
#         action = self.env.action_space.sample()  # Выбор случайного действия
#         obs, reward, terminated, info = self.env.step(action)
#         self.assertIsInstance(obs, np.ndarray, "Step should return an observation as a numpy array")
#         # self.assertIsInstance(reward, float, "Reward should be a float")
#         self.assertIsInstance(terminated, bool, "Terminated should be a boolean")
        
#     def test_max_steps(self):
#         """Checking that the episode ends when the maximum number of steps is reached"""
#         self.env.reset()
#         for _ in range(self.env.max_episode_steps - 1):
#             action = self.env.action_space.sample()
#             obs, reward, terminated, info = self.env.step(action)
#             self.assertFalse(terminated, "Episode should not terminate before max steps")
#         # last_step
#         action = self.env.action_space.sample()
#         obs, reward, terminated, info = self.env.step(action)
#         self.assertTrue(terminated, "Episode should terminate at max steps")
#         self.assertIn("reward", info, "Info should contain cumulative reward")
#         self.assertIn("length", info, "Info should contain episode length")

#     def test_seed(self):
#         """seed checking"""
#         obs1 = self.env.seed(42)
#         obs2 = self.env.seed(42)
#         np.testing.assert_array_equal(obs1, obs2, "Observations with the same seed should be equal")

#     def tearDown(self):
#         self.env.close()

# if __name__ == "__main__":
#     unittest.main()

import popgym 




env = popgym.envs.autoencode.AutoencodeEasy() #POPGymWrapper('AutoencodeEasy')
# print(env.deck.__dict__)
print(env.max_episode_length)
for i in range(53):
    action = env.action_space.sample()
    print(f'Action :{action}')
    print(env.step())