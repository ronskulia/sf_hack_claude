# Customer

This product is targeted towards municipal defense authorities.

Define the product as a decision-support tool: "Given your city&rsquo;s road network and your available air-defense assets, where should you place fixed units and stage mobile groups to maximize intercept probability against likely attack patterns?

The value proposition for customers is that they can run the simulation, and see which configuration achieves the maximal performance.



# Goal

To accurately model the adversarial strategies of drone attacks, at the level of a Ukrainian City

The goal is NOT for agents trained on one city to
generalise to another city (zero shot generalization is difficult). The goal is that the user can train the agents on any given configuration, and train the MARL
process on that given configuration to discover strategies/possible weaknesses.



# Targets

The attacking agent gains points from hitting targets.

Both agents know the number of points per target in advance.



# Agents

The gameplay will be controlled by two agents

The agents play in real-time.



## Attacking agent

Attacking agent can send out waves of drones.

The attacker pre-determines the trajectories of the drones. The attacker cannot alter the trajectory of a drone after that drone has been launched.

The attacker can choose to send any number of drones from any starting location, at any time.



## Defending agent

The defending agent controls technical mobile group units.

The mobile group units have to move, according to the topology of roads in the city

The defending agent should have fixed air-defense and mobile technical.
<https://en.wikipedia.org/wiki/Technical_(vehicle)>



### Mobile Group

The mobile groups move along roads. They must respect the roads.



### Fixed Units

The fixed units can be placed by the defending agent in advance of the round, but cannot
be moved during the round.



## Training of Agents

Agents will be trained by adversarial reinforcement learning.



# Possible Extensions (Outside of base project)

Extensions are not required in the core of the project,
but the architecture of the project should be flexible enough such that extensions could
be incorporated.


## Modeling Cities based on real-world Ukrainian topology

<https://www.openstreetmap.org/search?query=kyiv&zoom=5&minlon=-137.68066406250003&minlat=21.94304553343818&maxlon=-54.00878906250001&maxlat=51.17934297928929#map=11/50.4514/30.6210>

For the first iteration, just have one city. That city can be randomly generated.
The architecture of the program should be flexible enough to be extended to real-world cities.



## Having multiple round games, where drones + units are lost on each round



## Incorporating population density to minimize civilian causalities.



## Custom accuracy function of defending units as accuracy as function of distance.



## Cool isometric visualisation of scenario



## Uncertainty/Probability

At first, the defender should know with certainty the position of all drones that have been launched.

In later versions, the defender should only have a probability distribution/uncertainty/partial information about drones.



# Scenario

-   The city should contain roads, buildings, targets.
-   At first, do what is simplest (randomly generate a city), but leave architecture flexible enough so that users can import custom layouts of cities, including of real-life Ukrainian cities from OpenStreetMap.



# Tech Stack

Separate simulation engine from visualizer



## Simulation Engine

-   Python
-   PettingZoo (for agent RL)
-   JaxMARL (\`jax.vmap\` can run simulations in parallel)
-   OSMnx (for openstreetmaps)



## Visualizer

-   Visualizer connects to simulation engine via sockets.
-   Visualiser Three.js + React Three Fiber.
-   Visualizer is a client of the simulation logic (not involved in simulation logic). Visualizer recieves live information over websockets.
-   Can use FastAPI to glue together websockets.



## Experimentation Layer

-   The user should be able to run &ldquo;experiments&rdquo; with different configurations
-   The user should be able to export the results of all experiments to JSON to compare the results of different configurations.



# Level of Abstraction

-   continuous-space board game
-   Discrete tick simulation (Markov Decision Process)



# Codex's Memory Bank

I am Codex, an expert software engineer with a unique characteristic: my memory resets completely between sessions. This isn't a limitation - it's what drives me to maintain perfect documentation. After each reset, I rely ENTIRELY on my Memory Bank to understand the project and continue work effectively. 
You are not required to read all memory bank files, but read the files that you think 
might be helpful or provide useful information to you.

The memory bank is located at `./docs/*.md`

## Memory Bank Structure

The Memory Bank consists of core files and optional context files, all in Markdown format. Files build upon each other in a clear hierarchy:

### Core Files (Required)
 `projectbrief.md`
   - Foundation document that shapes all other files
   - Created at project start if it doesn't exist
   - Defines core requirements and goals
   - Source of truth for project scope

 `systemPatterns.md`
   - System architecture
   - Key technical decisions
   - Design patterns in use
   - Component relationships
   - Critical implementation paths

`techContext.md`
   - Technologies used
   - Development setup
   - Technical constraints
   - Dependencies
   - Tool usage patterns

### Additional Context
Create additional files/folders within `docs/*.md` when they help organize:
- Complex feature documentation
- Integration specifications
- API documentation
- Testing strategies
- Deployment procedures

You SHOULD create additional `docs/*.md`
files if you think there is something missing from the documentation that future versions of you would
like to know.

You should have files like:
- `docs/policy.md` to document the specific reward policy and training policy used.
- `docs/level_of_abstraction.md` to document what level of abstraction we are modelling and NOT modelling (i.e. not modelling aerodynamics, not modelling fuel consumption).

You SHOULD update existing `docs/*.md` files if you think those files are missing something/could be improved/
could be made more helpful.

# Git Tools
This repository is at 
https://github.com/ronskulia/sf_hack/

You have `git` and `gh` installed (and logged in). You may use them.
When you make a branch, do not include `codex` nor `claude` in the name of the branch.


# Nix tools
You may give yourself more tools by editing
`flake.nix`.
