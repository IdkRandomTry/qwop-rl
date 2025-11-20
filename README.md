# QWOP Reinforcement Learning Project

## Motivation
- Play qwop here > add link
- Controls are unintuitive for humans. Hence we find it hard to progress in first try. We didn't even properly understand what the buttons did. It was frustrating to play.
- Hence we believed it was a good game to test out knowledge of RL

## Gymnasium Wrapper for QWOP
- See github here > add link
- What is gymnasium
- What does the qwop wrapper do?
  We decided to explore the available qwop wrapper. You can see our experiments in the jupyter notebook labelled: [understanding_QWOP_gym.ipynb](My-RL/understanding_QWOP_gym.ipynb). We found it gave us access to the following
  - Environment
    - Browser Game
    - ChromeDriver
    - Lot of information 
  - Actions:
    - all combinations of Q,W,O,P
  - Default reward
    - ds/dt - dt > unpack

## Policy Gradient (Proximal Policy Optimization, an Actor Critic algorithm)
Why PPO? Continuous input and discrete output.

We used PPO by using the stable baseline library you can find the training cod [here](My-RL/playground.ipynb). It performed decently. You can see the result in the first cell of [showcase.ipynb](My-RL/showcase.ipynb). We noticed an interesting issue. We trained the model for some 'n' episodes. However due to the failure prone nature of the game, the model did not learn a lot from many episodes. Hence we decided to train the model for 'n' steps as a step (or a play) is a much more direct indicator of much a model has "learnt". We also implemented our own PPO algorithm. It performed better than the stable baseline's PPO reaching the finish line consistently! (NOTE: A likely reason is that we trained the custor PPO algorithm for longer :P). 

<video controls src="../../../../Users/Siddhesh/Downloads/QWOP-100m.mp4" title="QWOP-100m"></video>

## Double QN
Not done yet

## Deep QN 
Not done yet

## "Knee Scraping"
what is knee scraping? (a video)
A more stable but slow strategy.
Common in a lot of resources we referred.
What to do?

## Penalty for Low Torso
<video controls src="QWOP-Trying to Stand.mp4" title="Title"></video>
A "patch" for encouraging the model to learn to "stride" was to impose a penalty when the torso is below a threshold. The penalty was a ReLU function of the models torso_y level. This encouraged the model to stand up but resulted in early terminations and terrible starting moves (failure at ~0m)

## Custom Rewards
The above modification in the reward was a "patch" and did not go deep into understanding how the existing reward system works and if we could engineer a better reward system to encourage the model to stride. We noticed that the environment provided us with a lot off information which was not being used to its full potential. Hence we tried some custom reward functions!
-Still to do

## Future Direction
- A lot of resources we were refering faced the same problem of "knee scraping". Several of them turned to "Imitation learning" to encourage the model to learn to "stride". So that can be a promising future direction which will benefit the model to learn the fastest way to beat QWOP!