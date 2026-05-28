# Frozen Lake Problem
 
## Overview
 
Frozen Lake is a classic reinforcement learning problem from OpenAI's Gym environment. It presents an agent with a simple grid-world navigation task where the goal is to move from a starting position to a target location while avoiding hazards.
 
## Problem Description
 
### The Environment
 
The agent operates in a 4×4 grid (or optionally 8×8) representing a frozen lake. The grid contains four types of tiles:
 
- **S** - Start position (where the agent begins)
- **F** - Frozen tile (safe to walk on)
- **H** - Hole (hazard that ends the episode)
- **G** - Goal (destination to reach)
### The Task
 
The agent must navigate from the **Start (S)** position to the **Goal (G)** position while avoiding falling into any of the holes **(H)**. Each step on a frozen tile is safe, but stepping into a hole results in failure and episode termination.
 
### Example 4×4 Grid
 
```
SFFF
FHFH
FFFF
HFFG
```
 
In this layout:
- The agent starts at position (0,0)
- The goal is at position (3,3)
- Holes are scattered throughout and must be avoided
## Mechanics
 
### Actions
 
The agent can take one of four discrete actions at each step:
 
0. **Left** - Move west
1. **Down** - Move south
2. **Right** - Move east
3. **Up** - Move north
### Stochasticity
 
A key challenge in Frozen Lake is that the ice is slippery. When the agent takes an action, there's only a 1/3 chance of moving in the intended direction. There's a 1/3 chance of sliding left and a 1/3 chance of sliding right relative to the intended direction.
 
For example:
- Intend to go **Right** → 1/3 go Right, 1/3 go Up, 1/3 go Down
- Intend to go **Down** → 1/3 go Down, 1/3 go Left, 1/3 go Right
This stochasticity makes the problem more realistic and challenging than deterministic navigation.
 
### Rewards
 
- **+1** for reaching the Goal (G)
- **0** for all other transitions
- Episode terminates when reaching either the Goal or a Hole
## State Space
 
- **4×4 grid**: 16 states (one for each tile)
- **8×8 grid**: 64 states
- Each state is fully observable
