# ELSA: Robotics Use Case Benchmark
Development repository for the Robotics Use Case Benchmark

# Setup
Create a virtual environment or pyenv with `python==3.10.6`. Run:
```
$ pip install -r requirements.txt
```

# Train
To train with FL, run:
```
$ roscd elsa_tiago_fl
$ python3 src/elsa_tiago_fl/gym_fl_train.py -e [ENVIRONMENT] -m [MODEL] -n_workers [N. WORKERS]
```
With:

- `MODEL={dqn_new}`
- `ENVIRONMENT={tiago-v2_discrete}`
- `ENVIRONMENT {1,...,10}`

To modify the training parameters check `configs/models/[MODEL].yaml`.

For killing all parallel ROS-Gazebo simulations:
```
$ roscd gazebo_parallel
$ ./gazebo_parallel/src/simulations/kill_simulations.sh 
```