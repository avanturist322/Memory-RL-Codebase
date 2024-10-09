## POPGym: Benchmarking Partially Observable Reinforcement Learning

[Paper](https://arxiv.org/pdf/2303.01859.pdf)
[Code](https://github.com/proroklab/popgym)


**RepeatPrevious** - In this environment, both observation and action spaces are categorical with 4 discrete values. The task requires the agent to replicate an observation a_{t+k} = o_t that occurred k steps prior, at any given time step t + k. The agent at each time step needs to simultaneously remember k categorical values; accordingly, the value of k depends on the difficulty level, being 4 for Easy, 32 for Medium, and 64 for Hard, regardless of the fixed observation space size (which is 4).

**Autoencode** - 

На каждом шаге выдает один из элементов колоды 


Сначала агент получает наблюдение в виде двух чисел: mode - текущее состояние среды: play/watch, card_id - номер карты в колоде, длина эпизода = num_cards * 2 - 1, сначала агент на протяжении num_cards//2 шагов должен считать последовательность и запомнить ее, в течении этого времени награда за любое действие = 0, потом за следующие num_cards//2 шагов агент должен воспроизвести последовательность, в течении этого времени в качестве наблюдений всегда (0, 0),агент получает положительные/отрицательные награды за свои действия

Сложность определяет количество колод и тем самым длину запоминаемой последовательности

"""A game very similar to Simon, but backwards.

The agent receives a sequence of cards, and must output the cards it saw
in reverse order. E.g., seeing [1, 2, 3] means I should output them in the order
[3, 2, 1].
"""

During the WATCH phase, a deck of cards is shuffled and played
in sequence to the agent with the watch indicator set. The watch indicator is unset at the last
card in the sequence, where the agent must then output the sequence of cards in order. This tests
whether the agent can encode a series of observations into a latent state, then decode the latent
state one observation at a time


The environment operates in two distinct phases. In the “watch” phase, the environment generates a discrete observation from a set of 4 possible values at each time step; agent actions do not affect the observations during this phase. This phase spans the first half of the episode, lasting T /2 steps, where T is the total length of the episode. The episode lengths are set to 104, 208, and 312 steps for the Easy, Medium, and Hard difficulty levels, respectively. In the subsequent phase, the agent must recall and reproduce the observations by outputting corresponding actions, which are drawn from the 4-value space, similar to the observation space. Essentially, the first step in the second phase should be equal to the first observation in the “watch” phase, the subsequent action in the second phase ought to be the same as the second observation from phase one, and so on. The agent receives both a phase indicator and the categorical ID to be repeated in its observation. Note that at each time step, the agent must remember T /2 categorical values; thus, successful completion of this task requires an explicit memory mechanism, and any policy performing better than random chance is utilizing some form of memory.

**Concentration** -


A deck of cards is shuffled and spread out face down. The player
flips two cards at a time face up, receiving a reward if the flipped cards match. The agent must remember the value and position of previously flipped cards to improve the rate of successful matches.

At each step, the agent receives an observation consisting of multiple categories, each with N distinct values. The number of categories is 52 for Easy and Hard, and 104 for Medium. The values of N are 3 for Easy and Medium, and 14 for Hard. The task simulates a card game with a deck of cards spread face down. The agent can flip two cards at each step, and if they match, they remain face up; otherwise, they are turned face down again. The agent’s observation includes the state of the full deck, and the episode length corresponds to the minimal average number of steps required to solve the task optimally as determined by Morad et al. (2023). To outperform a random policy, the agent must remember card positions and values to find matches more efficiently. Even without memory, an agent can avoid flipping cards already turned face up, which conserves steps without yielding new information or reward, thereby outperforming random policies at a basic level. This is a marginal yet conspicuous improvement that shows a memory-less policy can gain over a random policy. Note that due to the episode length constraint, this task cannot be solved without memory


##### Environment Parameters

RepeatPrevious: 
* Easy: num_decks=1, k=4
* Medium: num_decks=2, k=32
* Hard: num_decks=3, k=64
Autoencode:
* Easy: num_decks=1
* Medium: num_decks=2
* Hard: num_decks=3
Concentration:





deck_type: String denoting what we are matching. Can be the card colors
(colors) or the card ranks (ranks)


class ConcentrationEasy(Concentration):
    def __init__(self):
        super().__init__(num_decks=1, deck_type="colors")


class ConcentrationMedium(Concentration):
    def __init__(self):
        super().__init__(num_decks=2, deck_type="colors")


class ConcentrationHard(Concentration):
    def __init__(self):
        super().__init__(num_decks=1, deck_type="ranks")



* Easy: 
* Medium
* Hard: 

RANKS = np.array(["a", "2", "3", "4", "5", "6", "7", "8", "9", "10", "j", "q", "k"])
SUITS = np.array(["s", "d", "c", "h"])
SUITS_UNICODE = ["♠", "♦", "♥", "♣"]
COLORS = np.array(["b", "r"])
DECK_SIZE = 52

RANKS and COLORS

Easy: context = 30 (2 вида карт)
Medium: context = 30 (2 вида карт)
Hard:  context = 60 (12 видов карт)