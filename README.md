# Deep Q-Learning Agent for Traffic Signal Control

A framework where a deep Q-Learning Reinforcement Learning agent tries to choose the correct traffic light phase at an intersection to maximize the traffic efficiency.

I have uploaded this here in order to help anyone that is searching for a good starter point for deep reinforcement learning with SUMO. This code is extracted from my master thesis and it represents a simplified version of the code used for my thesis work. I hope you can find this repository useful for your project.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. In my opinion, these are the easiest steps to follow in order to be able to run the algorithm starting from scratch. A computer with an NVIDA GPU is strongly recommended.

1. Download Anaconda ([official site](https://www.anaconda.com/distribution/#download-section)) and install.
2. Download SUMO ([official site](https://www.dlr.de/ts/en/desktopdefault.aspx/tabid-9883/16931_read-41000/) and install.
3. Follow [this](https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc) short guide in order to install tensorflow-gpu correctly and problem-free. Basically you need to open Anaconda Prompt for example and type the following commands and that's it:
```
conda create --name tf_gpu
activate tf_gpu
conda install tensorflow-gpu
```

I've used the following software versions: Python 3.6, SUMO traffic simulator 1.0.1, tensorflow 1.11.0

## Running the algorithm

- In order to run the algorithm, you need to run the file **tlcs_main.py** and the agent will start the training. You don't need to open any SUMO software, since everything it is loaded and done in the background. 

If you want to see the training process as it goes, you need to set to *True* the variable *gui* in the *TRAINING OPTIONS*, which is located at line 90 of **tlcs_main.py**. Keep in mind that viewing the simulation is very slow compared to the background training and you also need to close SUMO-GUI every time an episode ends, which is not practical.

When the agent ends the training, results will be stored under "*./model/model_xxxxx*". Results will include some graphs, the data used to generate the graphs and lastly the saved neural network.

**Training time:** ~45 seconds per episode, 1h 20min for 100 episodes, on a laptop equipped with i7-6700HQ, 16GB RAM, NVIDIA GTX 960M, SSD.

## The Deep Q-Learning Agent

**Framework**: Q-Learning with deep neural network.

**Context**: traffic signal control of 1 intersection.

**Environment**: a 4-way intersection with 4 incoming lanes and 4 outgoing lanes per arm. Each arm is 750 meters long. Each incoming lane defines the possible directions that a car can follow: left-most lane dedicated to lef-turn only; right-most lane dedicated to right-turn and straight; two middle lanes dedicated to only going straight. The layout of the traffic light system is as follows: the left-most lane has a dedicated traffic-light, while the others three lanes shares the same traffic light.

**Traffic generation**: For every episode, 1000 cars are created. The cars arrival timing are defined according to a Weibull distribution with shape 2 (fast increase of arrival until peak just before the mid-episode, then slow decreasing). 75% of vehicles spawned will go straight, 25% will turn left or right. Every vehicle have the same probability to be spawned at the beginning of every arm. On every episode the cars are generated randomly so is not possible to have two equivalent episode in term of vehicle's arrival layout.

**Agent ( Traffic Signal Control System - TLCS)**:
- **State**: discretization of incoming lanes into presence cells, which identify the presence or absence of at least 1 vehicle inside them. There are 20 cells per arm. 10 of them are placed along the left-most lane while the others 10 are placed in the others three lane. 80 cells in the whole intersection.
- **Action**: choiche of the traffic light phase from a 4 possible predetermined phases, which are the described below. Every phase has a duration of 10 seconds. When the phase changes, a yellow phase of 4 seconds is activated.
  - North-South Advance: green for lanes in the north and south arm dedicated to turn right or go straight.
  - North-South Left Advance: green for lanes in the north and south arm dedicated to turn left. 
  - East-West Advance: green for lanes in the east and west arm dedicated to turn right or go straight.
  - East-West Left Advance: green for lanes in the east and west arm dedicated to turn left. 
- **Reward**: change in *cumulative waiting time* between actions, where the waiting time of a car is the number of seconds spent with speed=0 since the spawn; *cumulative* means that every waiting time of every car located in an incoming lane is summed. When a car leaves an incoming lane (i.e. crossed the intersection), its waiting time is not considered anymore, therefore is a positive reward for the agent.
- **Learning mechanism**: the agent make use of the Q-learning equation *Q(s,a) = reward + gamma • max Q'(s',a')* to update the action values and a deep neural network to learn the state-action function. The neural network is fully connected with 80 neurons as input (the state), 5 hidden layers of 400 neurons each and the output layers with 4 neurons representing the 4 possible actions. Also, a mechanism of experience replay is implemented: the experience of the agent is stored in a memory and, at every step, a batch of randomized samples are extracted from the memory and used to train the neural network once the action values has been updated with the Q-learning equation.

## The code structure

The main file is **tlcs_main.py**. It basically handles the main loop that starts an episode on every iteration. At the end it saves the network and it also save 3 graphs: negative reward, cumulative delay and average queues. 

Overall the algorithm is divided into classes that handle different part of the training.
- The **Model** class is used to define everything about the deep neural network and it also contains some functions used to train the network and predict the outputs.
- The **Memory** class handle the memorization for the experience replay mechanism. A function is used to add a sample into the memory, while the other function retrieves a batch of samples from the memory.
- The **SimRunner** class handles the simulation. In particular, the function *run* allows the simulation of one episode. Also, some other functions are used during *run* in order to interact with SUMO, for example retrieving the state of the environment (*get_state*), set the next green light phase (*_set_green_phase*) or preprocess the data in order to train the neural network (*_replay*).
- The **TrafficGenerator** class contain the function dedicated to defining the route of every vehicle in one epsiode. The file created is *tlcs_train.rou.xml* which is placed in the "intersection" folder.

In the "intersection" folder there is one file called *tlcs.net.xml* which defines the structure of the environment, and it was created using SUMO NetEdit. The other file *tlcs_config_train.sumocfg* it is basically a linker between the environment file and the route file. 

## Author

* **Andrea Vidali** - *University of Milano-Bicocca*

If you need further information, I suggest you to look at my master thesis [here](https://www.dropbox.com/s/aqhdp0q6qhpx8q9/780747_Vidali_tesi.pdf?dl=0) or write me an e-mail at info@andreavidali.com.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

