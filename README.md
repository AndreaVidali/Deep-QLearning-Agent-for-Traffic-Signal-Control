# Deep Q-Learning Agent for Traffic Signal Control

A framework where a deep Q-Learning Reinforcement Learning agent tries to choose the correct traffic light phase at an intersection to maximize traffic efficiency.

I have uploaded this here to help anyone searching for a good starting point for deep reinforcement learning with SUMO. This code is extracted from my master thesis, and it represents a simplified version of the code used for my thesis work. I hope you can find this repository useful for your project.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine. In my opinion, the following are the easiest steps in order to run the algorithm from scratch, with the least amount of effort. A computer with an NVIDIA GPU is strongly recommended.

1. Download Anaconda ([official site](https://www.anaconda.com/distribution/#download-section)) and install.
2. Download SUMO ([official site](https://www.dlr.de/ts/en/desktopdefault.aspx/tabid-9883/16931_read-41000/)) and install.
3. Follow [this](https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc) short guide to install tensorflow-gpu correctly and problem-free. In short, the guide tells you to open Anaconda Prompt, or any terminal, and type the following commands:
```
conda create --name tf_gpu
activate tf_gpu
conda install tensorflow-gpu
```

I've used the following software versions: Python 3.7, SUMO traffic simulator 1.2.0, tensorflow 2.0

## Running the algorithm

1. Clone or download the repo.
2. Using the Anaconda prompt or any other terminal, navigate to the root folder and run the file **training_main.py** by executing:
```
python training_main.py
```

Now the agent should start the training.

You don't need to open any SUMO software since everything is loaded and done in the background. If you want to see the training process as it goes, you need to set to *True* the parameter *gui* contained in the file **training_settings.ini**. Keep in mind that viewing the simulation is very slow compared to the background training, and you also need to close SUMO-GUI every time an episode ends, which is not practical.

The file **training_settings.ini** contains all the different parameters used by the agent in the simulation. The default parameters aren't greatly optimized, so a bit of testing will likely increase the algorithm's current performance.

When the training ends, the results will be stored in "*./model/model_x/*" where *x* is an increasing integer starting from 1, generated automatically. Results will include some graphs, the data used to create the graphs, the trained neural network, and a copy of the ini file where the agent settings are.

Now you can finally test the trained agent. To do so, you have to run the file **testing_main.py**. The test involves a single episode of simulation, and the results of the test will be stored in "*./model/model_x/test/*" where *x* is the number of the model that you specified to test. The number of the model to test and other useful parameters are contained in the file **testing_settings.ini**.

**Training time:** ~27 seconds per episode, 45min for 100 episodes, on a computer equipped with i7-3770K, 8GB RAM, NVIDIA GTX 970, SSD.

## The code structure

The main file is **training_main.py**. It handles the main loop that starts an episode on every iteration. It also saves the network weights and three plots: negative reward, cumulative wait time, and average queues. 

Overall the algorithm is divided into classes that handle different parts of the training.
- The **Model** class defines everything about the deep neural network, and it also contains some functions used to train the network and predict the outputs. In the **model.py** file, two different **model** classes are defined: one used only during the training and only during the testing.
- The **Memory** class handle the memorization for the experience replay mechanism. A function adds a sample into the memory, while another function retrieves a batch of samples from the memory.
- The **Simulation** class handles the simulation. In particular, the function *run* allows the simulation of one episode. Also, other functions are used during *run* to interact with SUMO, for example: retrieving the state of the environment (*get_state*), set the next green light phase (*_set_green_phase*) or preprocess the data to train the neural network (*_replay*). Two files contain a slightly different **Simulation** class: **training_simulation.py** and **testing_simulation.py**. Which one is loaded depends if we are doing the training phase or the testing phase.
- The **TrafficGenerator** class contains the function dedicated to defining every vehicle's route in one episode. The file created is *episode_routes.rou.xml*, which is placed in the "intersection" folder.
- The **Visualization** class is just used for plotting data.
- The **utils.py** file contains some directory-related functions, such as automatically handling the creations of new model versions and the loading of existing models for testing.

In the "intersection" folder, there is a file called *environment.net.xml*, which defines the environment's structure, and it was created using SUMO NetEdit. The other file *sumo_config.sumocfg* it is a linker between the environment file and the route file.  

## The settings explained

The settings used during the training and contained in the file **training_settings.ini** are the following:
- **gui**: enable or disable the SUMO interface during the simulation.
- **total_episodes**: the number of episodes that are going to be run.
- **max_steps**: the duration of each episode, with 1 step = 1 second (default duration in SUMO).
- **n_cars_generated**: the number of cars that are generated during a single episode.
- **green_duration**: the duration in seconds of each green phase.
- **yellow_duration**: the duration in seconds of each yellow phase.
- **num_layers**: the number of hidden layers in the neural network.
- **width_layers**: the number of neurons per layer in the neural network.
- **batch_size**: the number of samples retrieved from the memory for each training iteration.
- **training_epochs**: the number of training iterations executed at the end of each episode.
- **learning_rate**: the learning rate defined for the neural network.
- **memory_size_min**: the min number of samples needed into the memory to enable the neural network training.
- **memory_size_max**: the max number of samples that the memory can contain.
- **num_states**: the size of the state of the env from the agent perspective (a change here also requires algorithm changes).
- **num_actions**: the number of possible actions (a change here also requires algorithm changes).
- **gamma**: the gamma parameter of the Bellman equation.
- **models_path_name**: the name of the folder that will contain the model versions and so the results. Useful to change when you want to group up some models specifying a recognizable name.
- **sumocfg_file_name**: the name of the .sumocfg file inside the *intersection* folder.

The settings used during the testing and contained in the file **testing_settings.ini** are the following (some of them have to be the same as the ones used in the relative training):
- **gui**: enable or disable the SUMO interface during the simulation.
- **max_steps**: the duration of the episode, with 1 step = 1 second (default duration in SUMO).
- **n_cars_generated**: the number of cars generated during the test episode.
- **episode_seed**: the random seed used for car generation (should not be a seed used during training).
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

**Environment**: a 4-way intersection with 4 incoming lanes and 4 outgoing lanes per arm. Each arm is 750 meters long. Each incoming lane defines the possible directions that a car can follow: left-most lane dedicated to left-turn only; right-most lane dedicated to right-turn and straight; two middle lanes dedicated to only going straight. The layout of the traffic light system is as follows: the left-most lane has a dedicated traffic-light, while the other three lanes share the same traffic light.

**Traffic generation**: For every episode, 1000 cars are created. The car arrival timing is defined according to a Weibull distribution with shape 2 (a rapid increase of arrival until the mid-episode, then slow decreasing). 75% of vehicles spawned will go straight, 25% will turn left or right. Every vehicle has the same probability of being spawned at the beginning of every arm. In every episode, the cars are randomly generated, so it is impossible to have two equivalent episodes regarding the vehicle's arrival layout.

**Agent ( Traffic Signal Control System - TLCS)**:
- **State**: discretization of oncoming lanes into presence cells, which identify the presence or absence of at least 1 vehicle inside them. There are 20 cells per arm. 10 of them are placed along the left-most lane while the other 10 are placed in the other three lanes. 80 cells in the whole intersection.
- **Action**: choice of the traffic light phase from 4 possible predetermined phases, described below. Every phase has a duration of 10 seconds. When the phase changes, a yellow phase of 4 seconds is activated.
  - North-South Advance: green for lanes in the north and south arm dedicated to turning right or going straight.
  - North-South Left Advance: green for lanes in the north and south arm dedicated to turning left. 
  - East-West Advance: green for lanes in the east and west arm dedicated to turning right or going straight.
  - East-West Left Advance: green for lanes in the east and west arm dedicated to turning left. 
- **Reward**: change in *cumulative waiting time* between actions, where the waiting time of a car is the number of seconds spent with speed=0 since the spawn; *cumulative* means that every waiting time of every car located in an incoming lane is summed. When a car leaves an oncoming lane (i.e. crossed the intersection), its waiting time is no longer counted. Therefore this translates to a positive reward for the agent.
- **Learning mechanism**: the agent make use of the Q-learning equation *Q(s,a) = reward + gamma • max Q'(s',a')* to update the action values and a deep neural network to learn the state-action function. The neural network is fully connected with 80 neurons as input (the state), 5 hidden layers of 400 neurons each, and the output layers with 4 neurons representing the 4 possible actions. Also, an experience replay mechanism is implemented: the experience of the agent is stored in a memory and, at the end of each episode, multiple batches of randomized samples are extracted from the memory and used to train the neural network, once the action values have been updated with the Q-learning equation.

## *Changelog - New version, updated on 12 Jan 2020*

- *Each training result is now stored in a folder structure, with each result being numbered with an increasing integer.*
- *New Test Mode: test the model versions you created by running a test episode with comparable results.*
- *Enabled a dynamic creation of the model by specifying, for each training, the width and the depth of the feedforward neural network that will be used.*
- *The neural network training is now executed at the end of each episode, instead of during the episode. This improves the overall speed of the algorithm.*
- *The code for the neural network is now written using Keras and Tensorflow 2.0.*
- *Added a settings file (.ini) for both training and testing.*
- *Added a minimum number of samples required into the memory to begin training.*
- *Improved code readability.*

## Author

* **Andrea Vidali** - *University of Milano-Bicocca*

If you need further information about the algorithm, I suggest you open an issue on the issues page.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Buy me a coffee!

Hi 👋 My name is Andrea.

If this repo helped you in some way and you want to say thanks, consider buying me a coffee!

<a href="https://www.buymeacoffee.com/andreavidali" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>
 

