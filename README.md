# Deep Q-Learning Agent for Traffic Signal Control

I have uploaded this here in order to help anyone that is searching for a good starter point for deep reinforcement learning with SUMO. You may want to take a look at **tlcs_train.py** and also read the description below.

**Framework**: Q-Learning with deep neural network.

**Context**: traffic signal control of an intersection.

**Environment**: a 4 way intersection with 4 incoming lanes per arm.

**Software**: Python 3.6, SUMO traffic simulator 1.0.1, tensorflow 1.11.0

**Agent ( Traffic Signal Control System - TLCS)**: the traffic light system that handle the incoming traffic.
- State: discretization of incoming lanes into presence cells, which identify the presence or absence of at least 1 vehicle inside them. There are 20 cells per arm, 80 cells in total.
- Action: choiche of the traffic light phase from a fixed set of predetermined phases, which they have a fixed duration of 10 seconds.
- Reward: change in cumulative waiting time between actions, where the waiting time of a car is the number of seconds spent with speed=0 since the spawn. When a car isn't in an incoming lane anymore (i.e. it crossed the intersection) the relative waiting time is not considered anymore.

The main file for training is **tlcs_train.py**. It is divided into three classes: Model, SimRunner and Memory. Moreover, the traffic generation is computed in the function _generate_routefile_.
The Model class is used to define everything about the deep neural network and it also contains some functions used to train the network and predict the outputs.
The SimRunner class is used to handle the simulation. Some functions are defined to divide semantically the agent's interactions with the simulator, such as retrieve the state, retrieve the waiting times or set the next action.
The Memory class is used to handle the experience replay algorithm used during the training.
In the Main function, the first little block of code is wrapped in an "OPTIONS" comment. These are the parameters that I usually change when I wanted to test different hyperparameters and see if that bring to better performance. The other code in the Main function just starts the training and handle the general training routine.

The results of the training are saved into the folder "model", created when the script ends. Consider that the training takes about 7 hours on my mid-high performance laptop.

In the "intersection" folder is defined the structure of the environment, created using SUMO NetEdit.

If you need further information, I suggest you to look at my master thesis [here](https://www.dropbox.com/s/aqhdp0q6qhpx8q9/780747_Vidali_tesi.pdf?dl=0) or write me an e-mail at info@andreavidali.com. This code is a slightly better version of the code I used for my thesis, but basically the main concepts are the same.
