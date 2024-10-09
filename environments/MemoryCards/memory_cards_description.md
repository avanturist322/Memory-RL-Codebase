## Memory Cards - Memory Card Game 

[Paper](https://arxiv.org/abs/2206.01078)
[Code](https://github.com/kevslinger/DTQN/blob/79ccf8b548a2f6263b770e051b42ced2932137ee/envs/memory_cards.py)

# Memory Cards

в колоде num_pairs пар карт, за один шаг можно переворачивать одну карту, агент должен найти все пары, +1 за правильно тогаданную пару, -1 за попытку открыть уже открытую пару, 0 за неверный выбор карты 

The agent is presented with N cards, and tries to find all the pairs.
    Each round the agent is shown one card, and has to select the card it thinks will pair it.
    Once a pair is found, it is removed from the pile.

    Consider the following episode:
        Say 1 represents a dog card, 2 Represents a cat card, 3 a bird card
        0 is hidden, and -1 means removed

    Reset:
    State: [1, 3, 1, 2, 2, 3]

    Obs:   [0, 0, 0, 2, 0, 0]
    Action: 2
    Reward: 0 # 0 reward for selecting incorrectly
    Obs: [0, 0, 0, 0, 0, 3]
    Action: 1
    Reward: 1 # Positive reward for selecting correctly
    Obs: [1, -1, 0, 0, 0, -1]
    Action: 2
    Reward: 1
    Obs: [-1, -1, -1, 0, 2, -1]
    Action: 0
    Reward: -1 # Negative reward for picking a removed card
    Obs: [-1, -1, -1, 0, 2, -1]
    Action: 3
    Reward: 1
    Obs: [-1, -1, -1, -1, -1, -1]
    Done: True

##### Environment Parameters

max_episode_steps=50
num_pairs - количество пар карт в колоде 