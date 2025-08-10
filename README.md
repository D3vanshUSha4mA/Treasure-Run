<div id="top"></div>

<!-- PROJECT SHIELDS -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links-->
<div align="center">

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

</div>

<!-- PROJECT LOGO -->
<br />
<!-- UPDATE -->
<div align="center">
  <a href="https://github.com/cgs-iitkgp/PROJECT_NAME">
     <img width="140" alt="image" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTLW7Aky6uVRhsyzD1aheY_xWsCUjJMEiE-lw&s">
  </a>

  <h3 align="center">Treasure-Run</h3>

  <p align="center">
  <!-- UPDATE -->
    <i>A Game combining classical algorithms with Reinforcement learning in enemies to improve the intelligence in enemy behaviour.
</i>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
<summary>Table of Contents</summary>

- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)
  - [Maintainer(s)](#maintainers)
  - [creators(s)](#creators)
- [Additional documentation](#additional-documentation)

</details>


<!-- ABOUT THE PROJECT -->
## About The Project
<!-- UPDATE -->
<div align="center">
  <a href="https://github.com/D3vanshUSha4mA/Treasure-Run">
    <img width="80%" alt="image" src="Screenshot 2025-08-10 225453.png">
  </a>
</div>

Treasure Run is a 2-D grid-based game built using pygame where-
The player collects the keys and reaches treasure.(Only one key is correct.)
Enemies patrol and chase using a combination of  DQN and A*. 

CORE COMPONENTS-
game.py-
Handles all the game logic - rendering,input,movement,health,collisions.   
rl_agent.py-
Defines the enemy agent using Deep-Q Learning with Pytorch.
main.py-
Runs the main loop, run the game from here.

WORKING-
The main function initializes Pygame and creates the Game object and calls the function game.update() every frame.
game.update()-
	This function runs every frame 
	It handles player input(handle_input())
	Animates movement(animate_player(),animate_enemies())
	Check if keys are collected or not
	Updates enemies behaviour using update_enemies()
	Triggers learning via enemy_agent.replay()
	Displays health,energy and grid
update_enemies():
	Each enemies decides what to do , if the player is visible chase it and keep chasing until the player goes out of a chase radius and otherwise patrol the grid using DQN , this improves the proper exploration of the grid.
Reward Policy-
Default=-1
If player is visible=+10
If enemy catches player=+50

Languages used- Python
Libraries used-
pygame,PyTorch…..

<p align="right">(<a href="#top">back to top</a>)</p>

## Getting Started

To set up a local instance of the application, follow the steps below.

### Prerequisites
The following dependencies are required to be installed for the project to function properly:
<!-- UPDATE -->
* npm
  ```sh
  npm install npm@latest -g
  ```

<p align="right">(<a href="#top">back to top</a>)</p>

### Installation

_Now that the environment has been set up and configured to properly compile and run the project, the next step is to install and configure the project locally on your system._
<!-- UPDATE -->
1. Clone the repository
   ```sh
   git clone https://github.com/D3vanshUSha4mA/Treasure-Run.git
   ```
2. Make the script executable
   ```sh
   cd ./D3vanshUSha4mA/Treasure-Run
   chmod +x ./D3vanshUSha4mA/Treasure-Run
   ```
3. Execute the script
   ```sh
   ./main.exe
   ```

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage
<!-- UPDATE -->
Use this space to show useful examples of how this project can be used. Additional screenshots, code examples and demos work well in this space.

<div align="center">
  <a href="https://github.com/cgs-iitkgp/PROJECT_NAME">
    <img width="80%" alt="image" src="">
  </a>
</div>

<p align="right">(<a href="#top">back to top</a>)</p>

## Contact

### Maintainer(s)

The currently active maintainer(s) of this project.

<!-- UPDATE -->
- [NAME](https://github.com/D3vanshUSha4mA)

### Creator(s)

Honoring the original creator(s) and ideator(s) of this project.

<!-- UPDATE -->
- [NAME](https://github.com/D3vanshUSha4mA)

<p align="right">(<a href="#top">back to top</a>)</p>

## Additional documentation

  - [License](/LICENSE)
  - [Code of Conduct](/.github/CODE_OF_CONDUCT.md)
  - [Security Policy](/.github/SECURITY.md)
  - [Contribution Guidelines](/.github/CONTRIBUTING.md)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->

[contributors-shield]: https://img.shields.io/github/contributors/D3vanshUSha4mA/Treasure-Run.svg?style=for-the-badge
[contributors-url]: https://github.com/D3vanshUSha4mA/Treasure-Run/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/D3vanshUSha4mA/Treasure-Run.svg?style=for-the-badge
[forks-url]: https://github.com/D3vanshUSha4mA/Treasure-Run/network/members
[stars-shield]: https://img.shields.io/github/stars/D3vanshUSha4mA/Treasure-Run.svg?style=for-the-badge
[stars-url]: https://github.com/D3vanshUSha4mA/Treasure-Run/stargazers
[issues-shield]: https://img.shields.io/github/issues/D3vanshUSha4mA/Treasure-Run.svg?style=for-the-badge
[issues-url]: https://github.com/D3vanshUSha4mA/Treasure-Run/issues
[license-shield]: https://img.shields.io/github/license/D3vanshUSha4mA/Treasure-Run.svg?style=for-the-badge
[license-url]: https://github.com/D3vanshUSha4mA/Treasure-Run/blob/master/LICENSE
