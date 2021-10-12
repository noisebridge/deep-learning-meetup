# deep-learning-meetup
Discussion and project code for the Deep Learning class

## Project ideas (feel free to add more!)

- Data pipeline tutorial with MNIST project (multiclass classification)
	Maybe 2 PT series with the first covering the initial model/data scaling and the second could get into stuff like balancing classes, logging training with tensorboard (works with pytorch or tf), viewing specific samples to see what the model is mislabelling etc.
- RNNs/LSTMs for time series data like stock price prediction (could cover backtesting, setting up a cuda environment etc.)
- RL through Kaggle competition (example: [Lux AI](https://www.kaggle.com/c/lux-ai-2021)
- RL through an [AI gym](https://gym.openai.com/) - could also look at [RL-baselines-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) for pretrained models and hack from there
- [Ray / RLlib](https://docs.ray.io/en/latest/rllib.html) lightweight framework for running RL on distributed systems
- [TFLite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers) deploying a model for something like wake word detection, object recognition etc (this would be getting into c++ a bit more)

## Shamelessly stealing this from a [reddit post](https://www.reddit.com/r/reinforcementlearning/comments/d0x8y6/deep_reinforcement_learning_roadmap):
I'm thinking we could skim the curriculum from some of these to come up with project ideas too

> There are three amazing lecture resources for learning RL. They all cover most if not every aspects of the field, but each has its focus.
> 
> David Silver's UCL RL Lectures focuses more on traditional planning/DP methods, such as policy/value iteration, MC, TD, bandit problem. Lecturer comes from a game playing background. His lecture closely mirrors Sutton & Barto's RL textbook. It's THE textbook in the field, and I recommend it. The lecturer also worked on AlphaGo, so he has a few lectures highlighting that as well which was pretty cool. He does not go over the frontiers of RL research much. (Inverse RL, Meta learning, etc)
> 
> Berkeley's RL class, taught by Sergey Levine, focuses on more modern RL methods which heavily uses deep neural nets as function approximator. Naturally, policy gradient based methods are given emphasis, and a lot of cutting edge research topics are covered. Lecturer comes from a robotics background.
> 
> Stanford's RL class taught by Emma Brunskill, is a nice balance between the two, and is the one that I recommend the most for beginners. Does a great job covering foundational RL like Silver, but also covers modern methods like Levine. Has some lecturers covering frontier research, but not as much as Levine's class. Lecturer comes from a more diverse background: healthcare, education, etc.`
