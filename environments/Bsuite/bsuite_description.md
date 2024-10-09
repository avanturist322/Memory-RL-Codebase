## Behaviour Suite for Reinforcement Learning

[Paper](https://arxiv.org/abs/1908.03568)
[Code](https://github.com/google-deepmind/bsuite)


**Bsuite** 

**MemoryLength:**

на каждом шаге наблюдение - либо +1 либо -1, награда дается только на последнем шаге, если агент правильно предскажет i-ое значение в начальном наблюдении, i-ое значение задается на последнем шаге в obs[1]

it is designed to test the number of sequential
steps an agent can remember a single bit. The underlying environment is based on a stylized T-maze (O’Keefe & Dostrovsky, 1971), parameterized by a length N ∈ N. Each episode lasts N steps with observation ot = (ct, t/N ) for t = 1, .., N and action space A = {−1, +1}. The context c1 ∼ Unif(A) and ct = 0 for all t ≥ 2. The reward rt = 0 for all t < N , but rN = Sign(aN = c1). For the bsuite experiment we run the agent on sizes N = 1, .., 100 exponentially spaced and look at the average regret compared to optimal after 10k episodes. The summary ‘score’ is the percentage of runs for which the average regret is less than 75%
of that achieved by a uniformly random policy.


**DiscountingChain:**

агент может выбрать одно из k действий, каждое из действий соответствует цепочке, в которой агент получает награду только на N-ом шаге (до и после награда - 0) для какого-то определенного действия из k возможных награда будет больше на 10%, чем для остальных - агент должен научиться определять такое действие


Observation is two pixels: (context, time_to_live)

Context will only be -1 in the first step, then equal to the action selected in
the first step. For all future decisions the agent is in a "chain" for that
action. Reward of +1 come  at one of: 1, 3, 10, 30, 100

However, depending on the seed, one of these chains has a 10% bonus.


##### Environment Parameters

**MemoryLength:** 

obs = [time, query, num_bits of context]

memory_length - длина эпизода и по совместительству длина последовательности, которую модель должна обработать, запомнив первое наблюдение
num_bits - размерность вектора наблюдений - 2


**DiscountingChain:**

число возможных действий: 5, шаги, на которых агент получить награды для каждого из действий: [1, 3, 10, 30, 100]

mapping_seed - индекс цепочки, в которой агент получает повышенную награду (mapping_seed = mapping_seed % self._n_actions)