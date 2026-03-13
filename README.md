<p align="center">
 <img height=350px src="./simulation-output.png" alt="Simulation output">
</p>

<h1 align="center">Basic Traffic Intersection Simulation</h1>

<div align="center">

[![Python version](https://img.shields.io/badge/python-3.1+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

<h4>A simulation developed from scratch using Pygame to simulate the movement of vehicles across a traffic intersection having traffic lights with a timer.</h4>

</div>

-----------------------------------------
### Description

* It contains a 4-way traffic intersection with traffic signals controlling the flow of traffic in each direction. 
* Each signal has a timer on top of it which shows the time remaining for the signal to switch from green to yellow, yellow to red, or red to green. 
* Vehicles such as cars, bikes, buses, and trucks are generated, and their movement is controlled according to the signals and the vehicles around them. 
* This simulation can be further used for data analysis or to visualize AI or ML applications. 

### Prerequisites

[Python 3.1+](https://www.python.org/downloads/)

------------------------------------------
### Installation

 * Step I: Clone the Repository
```sh
      $ git clone https://github.com/mihir-m-gandhi/Basic-Traffic-Intersection-Simulation
```
  * Step II: Install the required packages
```sh
      # On the terminal, move into Basic-Traffic-Intersection-Simulation directory
      $ cd Basic-Traffic-Intersection-Simulation
      $ pip install pygame
```
* Step III: Run the code
```sh
      # To run simulation
      $ python simulation.py
```

------------------------------------------
### Author

Mihir Gandhi - [mihir-m-gandhi](https://github.com/mihir-m-gandhi)

------------------------------------------
### License
This project is licensed under the MIT - see the [LICENSE](./LICENSE) file for details.

You can naviagte through the simulation using the a, s, w, and d keys. This will let you look at the different intersections.

### RL State Bucket Profiles

For tabular RL (`--mode rl`), you can control state discretization with:

- `--rl-state-profile coarse`
- `--rl-state-profile default`
- `--rl-state-profile fine`

The RL state remains:

- `(cur_phase, dom_q, dom_wait, q_bucket, max_wait_bucket)`

Only the queue/wait bucket granularity changes. `fine` gives more resolution (more states), while `coarse` is faster but less precise.

Example headless RL train run with finer buckets:

```bash
python simulation.py --headless --mode rl --rl-train --seed 42 --spawn-rate 2.0 --duration 300 --rl-state-profile fine
```
