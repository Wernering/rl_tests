# Description:

This repository was made to visualize the steps along the solution of the Gridworld problem proposed in Chapter 6 of "Reinforcement Learning: An Introduction" from Richard Sutton and Andrew Barto.

The Objective of the problem is for the agent (green box) to arrive to a specific coordinate (orange box) in the fewest steps possible (shortest route) while there is wind that reroute the character.

There is 2 ways to execute the code. with and without UI.

In the file "algorithm.py" are the classes for the AI logic. You can run this file as __main__ to be able the final results of the iterations. A Graph will be made after the execution finish.

In the "game.py" is the pygame Class that allows the user to visualize all the steps of the simulation.

# Configurations

The ways of the simulation can be changed i the "game_config.py" file. There is possible to adjust:
 - Window Size
 - Grid Size
 - Columns and rows of the grid
 - Number of movements allowed
 - Epsilon
 - Alpha
 - TD Method (Between SARSA, EX-SARSA and Q-LEARN)
 - Wind configuration, for columns and rows

 # Controls

 When running the simulation with the UI, it is possible to control some behaviours of the game:
 - SPACE_BAR: Pause / Continue
 - UP: Speed Up the simuation (fps)
 - DOWN: Slow Down the simulation
 - M: change view between state-action matrix and wind configuration
 - W: Change the view of the wind (if activated) between arrows and vectors (row, cols)
 - G: set epsilon to 0 or EPSILON. Deactivate the exploration factor
 - T: Shows the best route discovered in blue Lines
 - Ctrl + S: Pause the simulation and generate Graph of results until that iteration
