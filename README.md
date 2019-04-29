# Deep Q-Learning Agent for Traffic Signal Control

I have uploaded this here in order to help anyone that is searching for a good starter point for deep reinforcement learning with SUMO. You may want to take a look at **tlcs_train.py** and also read the description below.

**Framework**: Q-Learning with deep neural network.

**Context**: traffic signal control of an intersection.

**Environment**: a 4 way intersection wit 4 incoming lanes per arm.

**Versions**: Python 3.6, SUMO traffic simulator 1.0.1, tensorflow 1.11.0

**Agent ( Traffic Signal Control System - TLCS)**: the traffic light system that handle the incoming traffic.
- State: discretization of incoming lanes into presence cells. 80 cells per intersection arm.
- Action: choiche of the traffic light phase with fixed duration.
- Reward: change in cumulative waiting time between actions.

The main file for training is **tlcs_train.py**. It is divided into three classes: Model, SimRunner and Memory. Also, the generation's timing of cars are defined in the function generate_routefile.
The Model class is used to define everything about the deep neural network and it also contains some functions used to train the network and predict the outputs.
The SimRunner class is used to handle the simulation. Some functions are defined to divide semantically the agent's interactions with the simulator.
The Memory class is used to handle the experience replay algorithm used during the training.
In the Main function, the first little block of code is wrapped in an "OPTIONS" comment. These are the parameters that I usually change when I wanted to test different agent's hyperparameters and see if that bring to better performance. The other code in the Main function just starts the training and handle this process.

The results of the training are saved into the folder "model", created when the script ends. Consider that the training takes about 7 hours on my laptop.

In the "intersection" folder is defined the structure of the environment, created using SUMO NetEdit.

If you need further information, I suggest you to look at my master thesis [here](https://www.dropbox.com/s/aqhdp0q6qhpx8q9/780747_Vidali_tesi.pdf?dl=0) or write me an e-mail at info@andreavidali.com. This code is a slightly better version of the code I used for my thesis, but basically the main concepts are the same.
