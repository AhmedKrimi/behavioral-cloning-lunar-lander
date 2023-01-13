# Behavioral Cloning Lunar Lander

## Project Description

Lunar Lander is an OpenAIGym Environment whose goal is to navigate a space-ship and make it land
between the two yellow flags with up right position and minimum landing velocity. As shown in the figure,
there are four discrete action classes: upwards, left, right and no-action.
In this project, we approach this task through imitation learning, where a human plays the game and feeds
the training data so that the agent can learn from it, thereby reformulating the problem as a supervised
learning problem.

## Getting started

Besides Pytorch, OpenAI Gym and Tensorboard are needed to be installed
`````
git clone https://github.com/openai/gym.git
cd gym
pip install -e ’.[box2d]’
`````
## Data collection

Data can be collected by running the following command and data will be stored every 2500 steps in ````./data````
`````
fly manually.py --collect data
`````

## Agent

The definition of the agent imitating the policy of the human playing the game can be found in  ````./agent````

## Poster

<p align="center">
  <img src="./poster/poster_1.jpg">
</p>

<p align="center">
  <img src="./poster/poster_2.jpg">
</p>