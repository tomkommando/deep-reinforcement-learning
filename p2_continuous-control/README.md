# Reacher Project

In a [Unity ML](https://github.com/Unity-Technologies/ml-agents) environment, we train a double-jointed arm `agent` to reach moving objects.  A reward of `+0.1` is provided for each step that the agent's hand is in the goal location.  The goal is to maintain its position at  the target location for as many time steps as possible. The task is episodic and the environment is considered solved when the agent manages to score `+30` on average over `100` consecutive episodes.

The observation space consists of `33` variables  corresponding to position, rotation, velocity, and angular velocities of the arm.  Each `action` is a vector with four numbers, corresponding to  torque applicable to two joints.  Every entry in the action vector must  be a number between `-1` and `1`.

We train the agent using [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971) algorithm and the training process is documented [here](.\Report.md)

### Getting Started

To run the code, you need a Python 3.6 environment with required dependencies installed.

1. Create environment

```
conda create --name reacherproject python=3.6
source activate reacherproject
```


2. Clone this repository and install requirements

```
git clone https://github.com/tomkommando/ReacherProject.git
cd ReacherProject
pip install -r requirements.txt
```

3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

- **_Version 1: One (1) Agent_**
  - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
  - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
  - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
  - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

- **_Version 2: Twenty (20) Agents_**
  - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
  - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
  - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
  - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)


4. Hyperparameter settings can be tweaked  in `parameters.py`.

5. Train the agent by running `train_agent.py` 

```
python train_agent.py
```

6. if you want to watch trained agent, follow the instructions in  `train_agent.py`. Saved weights for trained agent can be found in the `output\` folder.