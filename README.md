# Swarming bats and the collective soundscape: replication and extension

Course project for *Collective Behaviour*.

## Team

| Name | GitHub account |
|------|----------------|
| Aleksandra Zec | [@aleksandrazec](https://github.com/aleksandrazec) |
| Matej Kunovski | [@Matej3334](https://github.com/Matej3334) |
| Jurij Grohar | [@jgrohar](https://github.com/jgrohar) |

## Project description

We will recreate the agent-based model of Vanderelst & Peremans (2025), in which echolocating bats (*Tadarida brasiliensis*) fly through corridors in dense swarms and avoid walls and each other without isolating their own echoes. Each simulated bat only compares the loudness of the overall soundscape (calls of others, body echoes, wall echoes, and their secondary echoes) between its left and right ear and turns away from the louder side. We will reimplement this model independently in Python, reproduce the paper's main results (collision counts, wall and inter-agent distance distributions, the effect of perched bats, and scaling from 25 to 50 agents), and then extend it. The paper itself notes that its agents spread out more than real bats because no acoustic mechanism keeps the swarm together. Our main extension is therefore to add a cohesion mechanism on top of the acoustic repulsion, and to study the trade-off between group cohesion, collision avoidance and exit throughput. In addition, we will run systematic sensitivity analyses (swarm density, call rate, flight speed, turning limit, target strengths, number of agents) to see how robust the collective soundscape strategy is.

## Starting point

- **Paper:** Vanderelst D, Peremans H (2025). *How swarming bats can use the collective soundscape for obstacle avoidance.* PLOS Computational Biology 21(5): e1013013. https://doi.org/10.1371/journal.pcbi.1013013
- **Original code and data:** https://osf.io/qsm8c/ (DOI: 10.17605/OSF.IO/QSM8C). We use it as a reference to check our own implementation, not as the basis of our code.
- **What we will build:** a simulation of a group of simple agents flying through corridors (straight, curved, branching). Each agent senses how loud the surroundings are on its left and right, and steers away from the louder side. Sound gets quieter with distance, and walls and other agents reflect sound.
- **How we will implement it:**
  - Start with a very simple version (agents, walls, steering) and check that the agents move through the corridor.
  - Add the sound model step by step, first in a simplified form, and add more realistic details only if time allows.
  - Run the same experiments and measurements as the paper (collisions, distances to walls and to other bats) and compare our results to theirs.
- **How we will extend it:** the paper shows that its bats spread out more than real bats and have no way of staying together. We will add a simple rule that keeps the group together (for example, a tendency to follow nearby neighbours) and study how this changes collisions and how quickly the group gets out. We will also test how sensitive the results are to the main settings, such as group size, density and flight speed.
- **Tools:** Python, with the code and results kept in this repository so everything can be reproduced.

## Reference

Vanderelst D, Peremans H. How swarming bats can use the collective soundscape for obstacle avoidance. *PLoS Comput Biol.* 2025;21(5):e1013013. https://doi.org/10.1371/journal.pcbi.1013013
