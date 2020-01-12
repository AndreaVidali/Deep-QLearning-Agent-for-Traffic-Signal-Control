# Deep Q-Learning Agent for Traffic Signal Control

A framework where a deep Q-Learning Reinforcement Learning agent tries to choose the correct traffic light phase at an intersection to maximize the traffic efficiency.

I have uploaded this here in order to help anyone that is searching for a good starter point for deep reinforcement learning with SUMO. This code is extracted from my master thesis and it represents a simplified version of the code used for my thesis work. I hope you can find this repository useful for your project.

## Improved version - 12 Jan 2020

Changelog:
- Each traning result is now stored in a folder structure, with each result being numbered with an increasing integer.
- New Test Mode: test the model versions you created by running a test episode with comparable results.
- Enabled a dynamic creation of the model by specifying, for each training, the width and the depth of the feedforward neural network the is going to be used.
- The training of the neural network is now executed at the end of each episode, instead of during the episode. This improve the overall speed of the algorithm.
- The code for the neural network is now written using Keras and Tensorflow 2.0.
- Added a settigs file (.ini) for both training and testing.
- Improved the tracking of the durations of both simulation and training of each episode.
- Added a minimum number of samples required into the memory in order to begin training.
- Improved the naming and the usage of the variables dedicated to save the stats of the episode.
- Improved docstrings of functions.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. In my opinion, these are the easiest steps to follow in order to be able to run the algorithm starting from scratch. A computer with an NVIDA GPU is strongly recommended.

1. Download Anaconda ([official site](https://www.anaconda.com/distribution/#download-section)) and install.
2. Download SUMO ([official site](https://www.dlr.de/ts/en/desktopdefault.aspx/tabid-9883/16931_read-41000/)) and install.
3. Follow [this](https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc) short guide in order to install tensorflow-gpu correctly and problem-free. Basically the guide tells you to open Anaconda Prompt, or any terminal, and type the following commands:
```
conda create --name tf_gpu
activate tf_gpu
conda install tensorflow-gpu
```

I've used the following software versions: Python 3.7, SUMO traffic simulator 1.2.0, tensorflow 2.0

## Running the algorithm

- Now you are ready to run the algorithm. To do so, you need to run the file **training_main.py** by executing the following simple command on the Anaconda prompt or any other terminal and the agent will start the training:
```
python training_main.py
```

You don't need to open any SUMO software, since everything it is loaded and done in the background. If you want to see the training process as it goes, you need to set to *True* the parameter *gui* contained in the file **training_settings.ini**. Keep in mind that viewing the simulation is very slow compared to the background training and you also need to close SUMO-GUI every time an episode ends, which is not practical.

The file **training_settings.ini** contains all the different parameters used by the agent in the simulation. The defualt parameters are not that optimized, so is very likely that a bit of testing will increase the current performance of the agent.

When the training ends, the results will be stored in "*./model/model_x/*" where *x* is an increasing integer starting from 1, generated automatically. Results will include some graphs, the data used to generate the graphs, the trained neural network and a copy of the ini file where the agent settings are.

Now you can finally test the trained agent. To do so, you will run the file **testing_main.py**. The test involve a single episode of simulation, and the results of the test will be stored in "*./model/model_x/test/*" where *x* is the number of the model that you specified to test. The number of the model to test and other useful parameters are contained in the file **testing_settings.ini**.

**Training time:** ~27 seconds per episode, 45min for 100 episodes, on a computer equipped with i7-3770K, 8GB RAM, NVIDIA GTX 970, SSD.

## The code structure

The main file is **training_main.py**. It basically handles the main loop that starts an episode on every iteration. At the end it saves the network and it also save 3 graphs: negative reward, cumulative wait time and average queues. 

Overall the algorithm is divided into classes that handle different part of the training.
- The **Model** class is used to define everything about the deep neural network and it also contains some functions used to train the network and predict the outputs. In the **model.py** file, two different **model** classes are defined: one used only during the training, one used only during the testing.
- The **Memory** class handle the memorization for the experience replay mechanism. A function is used to add a sample into the memory, while another function retrieves a batch of samples from the memory.
- The **Simulation** class handles the simulation. In particular, the function *run* allows the simulation of one episode. Also, some other functions are used during *run* in order to interact with SUMO, for example retrieving the state of the environment (*get_state*), set the next green light phase (*_set_green_phase*) or preprocess the data in order to train the neural network (*_replay*). There are two files that contain a slightly different **Simulation** class: **training_simulation.py** and **testing_simulation.py**. Which one is loaded depends of course if we are doing the training phase or the testing phase.
- The **TrafficGenerator** class contain the function dedicated to defining the route of every vehicle in one epsiode. The file created is *episode_routes.rou.xml* which is placed in the "intersection" folder.
- The **Visualization** class is just used for plotting data.
- In the **utils.py** file are contained some directory-related functions, such as automatically handle the creations of new model versions and the loading of existing models for the testing.

In the "intersection" folder there is a file called *environment.net.xml* which defines the structure of the environment, and it was created using SUMO NetEdit. The other file *sumo_config.sumocfg* it is basically a linker between the environment file and the route file. 

## The settings explained

The settings used during the training and contained in the file **training_settings.ini** are the following:
- **gui**: enable or disable the SUMO interface during the simulation.
- **total_episodes**: the number of episodes that are going to be run.
- **max_steps**: the duration of each episode, with 1 step = 1 second (default duration in SUMO).
- **n_cars_generated**: the number of cars that are generated during a single episode.
- **green_duration**: the duration in seconds of each green phase.
- **yellow_duration**: the duration in seconds of each yellow phase.
- **num_layers**: the number of hidden layers in the neural network.
- **width_layers**: the number of neuron per layer in the neural network.
- **batch_size**: the number of samples retrieved from the memory for each training iteration.
- **training_epochs**: the number of training iterations executed at the end of each episode.
- **learning_rate**: the learning rate defined for the neural network.
- **memory_size_min**: the min number of samples needed into the memory to enable the training of the neural network.
- **memory_size_max**: the max number of samples that the memory can contain.
- **num_states**: the size of the state of the env from the agent perspective (a change here requires also algorithm changes).
- **num_actions**: the number of possible actions (a change here requires also algorithm changes).
- **gamma**: the gamma parameter of the bellman equation.
- **models_path_name**: the name of the folder that will contains the model versions and so the results. Useful to change when you want to group up some models specifying a recognizable name.
- **sumocfg_file_name**: the name of the .sumocfg file inside the *intersection* folder.

The settings used during the testing and contained in the file **testing_settings.ini** are the following (some of them have to be the same of the ones used in the relative training):
- **gui**: enable or disable the SUMO interface during the simulation.
- **max_steps**: the duration of the episode, with 1 step = 1 second (default duration in SUMO).
- **n_cars_generated**: the number of cars generated during the test episode.
- **episode_seed**: the random seed used for car generation, that should be a seed not used during training.
- **green_duration**: the duration in seconds of each green phase.
- **yellow_duration**: the duration in seconds of each yellow phase.
- **num_states**: the size of the state of the env from the agent perspective (same as training).
- **num_actions**: the number of possible actions (same as training).
- **models_path_name**: The name of the folder where to search for the specified model version to load.
- **sumocfg_file_name**: the name of the .sumocfg file inside the *intersection* folder.
- **model_to_test**: the version of the model to load for the test. 

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
- **Learning mechanism**: the agent make use of the Q-learning equation *Q(s,a) = reward + gamma • max Q'(s',a')* to update the action values and a deep neural network to learn the state-action function. The neural network is fully connected with 80 neurons as input (the state), 5 hidden layers of 400 neurons each and the output layers with 4 neurons representing the 4 possible actions. Also, a mechanism of experience replay is implemented: the experience of the agent is stored in a memory and, at the end of each episode, multiple batches of randomized samples are extracted from the memory and used to train the neural network once the action values has been updated with the Q-learning equation.

## Author

* **Andrea Vidali** - *University of Milano-Bicocca*

If you need further information, I suggest you to look at my master thesis [here](https://www.dropbox.com/s/aqhdp0q6qhpx8q9/780747_Vidali_tesi.pdf?dl=0) or write me an e-mail at info@andreavidali.com.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

