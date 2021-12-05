For more information, please refer to the [Sutton and Barto book](http://incompleteideas.net/book/the-book-2nd.html).

# Training a Lunar Lander environment


This is example code for training an RL AI to successfully complete the Lunar Lander OpenAI gym environment. You'll need to install the environment in environment.yml, if you don't have Conda it should work through the gym library and tensorflow.

From a most recent test, this should successfully land and get over 200 points a majority of the time after training (I got an average score of 153 after 1500 episodes). I encourage people to mess with different arguments, particularly the `gamma` argument.

## What is this?

This is an adaptation of the Double Deep Q network model for the Lunar Lander. Essentially, it trains a small fully connected neural network to determine the Q value of each action given the state as input. This takes several hours to train in completion on a standard laptop. 

I also included some of the report I wrote for the original project. I found it really interesting to work on it, and hope you do too. Please don't share this widely, especially to someone who is or will be taking the clRL class at Georgia Tech. 

## How to run the Lunar Lander

From the help file:

```
usage: lunar_lander_trainer.py [-h] [--gamma GAMMA] [--episodes EPISODES]
                               [--C C] [--learning_rate LEARNING_RATE]
                               [--initial_epsilon INITIAL_EPSILON]
                               [--final_epsilon FINAL_EPSILON]
                               [--replay_memory_size REPLAY_MEMORY_SIZE]

Train a Lunar Lander model.

optional arguments:
  -h, --help            show this help message and exit
  --gamma GAMMA         Gamma value
  --episodes EPISODES
  --C C
  --learning_rate LEARNING_RATE
  --initial_epsilon INITIAL_EPSILON
  --final_epsilon FINAL_EPSILON
  --replay_memory_size REPLAY_MEMORY_SIZE
```


## TODO

I'm going to add a Tensorboard printer to make tracking the progress as it trains easier.
