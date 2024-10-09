## Passive T-maze-flag  

[Paper](https://arxiv.org/abs/2307.03864)
[Code](https://github.com/twni2016/Memory-RL)

##### Memory Cards

Агент появляется в начале коридора и наблюдает подсказку. Далее агент должен идти по прямо по коридору до развилки. Перед развилкой агент получает флаг, сигнализирующий, что на следующем шаге нужно будет повернуть. При правильном повороте агент получает награду.


To investigate agent’s long-term memory on very long environments (the inference
trajectory length is much longer than the context length K) we used a modified version of the T-Maze environment [35]. The agent’s objective in this environment is to navigate from the beginning of the T-shaped maze to the junction and choose the correct direction, based on a signal given at the beginning of the trajectory using four possible actions a ∈ {lef t, up, right, down}. This signal, represented as the clue variable and equals to zero everywhere except the first observation, dictates whether the agent should turn up (clue = 1) or down (clue = −1). Additionally, a constraint on the episode duration T = L + 2, where the maximum duration is determined by the length of the corridor L to the junction, adds complexity to the problem. To address this, a binary flag, represented as the f lag variable, which is equal to 1 one step before the junction and 0 otherwise, indicating
the arrival of the agent at the junction, is included in the observation vector. Additionally, a noise channel is added to the observation vector, with random integer values from the set {−1, 0, +1}. The observation vector is thus defined as o = [y, clue, f lag, noise], where y represents the vertical coordinate. The reward r is given only at the end of the episode and depends on the correctness of the agent’s turn at the junction, being 1 for a correct turn and 0 otherwise. This formulation deviates
from the traditional Passive T-Maze environment [35] (different observations and reward functions) and presents a more intricate set of conditions for the agent to navigate and learn within the given
time constraint.

##### Environment Parameters

1. episode_timeout - длительность эпизода
2. corridor_length  - длина коридора до развилки (по умолчанию corridor_length = episode_timeout - 2)
3. goal_reward  - награда при правильном повороте
4. penalty - штраф за каждый шаг (по умолчанию penalty = 0), но если нужен непрерывный сигнал, то можно установить penalty = -1 / (episode_timeout - 1)