<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Kyppy/lstm_load_forecast">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
  
<h3 align="center">LSTM Time-series Electrical Load Prediction</h3>

  <p align="center">
    A long short-term memory (LSTM) time-series prediction model to forecast residential power consumption. The LSTM models are trained using electrical data measurements from a real home.
    <br />
    <a href="https://github.com/Kyppy/lstm_load_forecast"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Kyppy/lstm_load_forecast">View Demo</a>
    ·
    <a href="https://github.com/Kyppy/lstm_load_forecast/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/Kyppy/lstm_load_forecast/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

In this project a LSTM neural network is used to produce a time-series prediction model that forecasts residential power consumption. The models are trained on a <a href="https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption">publicly available</a> set of residential power measurements from a real home using a Tensorflow-Keras backend. The LSTM uses a 3-layer, stacked architecture.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Tensorflow][TF]][TF-url]
* [![Keras][KR]][KR-url]
  
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy of this project running follow these steps.

### Prerequisites

* Python
  ```python
  version 3.10.8
  ```

  ### Installation
  
1. Clone the repo
   ```sh
   git clone https://github.com/Kyppy/lstm_load_forecast.git
   ```
2. Install required packages
 * Using Pip
   ```sh
   pip install -r pip_requirements.txt
   ```
 * Using Conda
   ```sh
   conda install --yes --file conda_requirements.txt
   ```
3. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin github_username/repo_name
   git remote -v # confirm the changes
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/github_username/repo_name/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Kyppy Smani - simanikyppy@gmail.com

Project Link: [https://github.com/Kyppy/lstm_load_forecast](https://github.com/Kyppy/lstm_load_forecast)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- RESEARCH PUBLICATIONS -->
## Research Publications

The following peer-reviewed research papers were published with findings produced using code from this project.

* [“Using LSTM To Perform Load Modelling For Residential
Demand Side Management”](https://ieeexplore.ieee.org/document/10057875)
* [“Using LSTM To Perform Load Predictions For Grid-
Interactive Buildings”](https://www.researchgate.net/publication/381076004_Using_LSTM_to_Perform_Load_Predictions_for_Grid-Interactive_Buildings)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[KR]: https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white
[KR-url]: https://keras.io/
[license-shield]: https://img.shields.io/github/license/Kyppy/lstm_load_forecast.svg?style=for-the-badge
[license-url]: https://github.com/Kyppy/lstm_load_forecast/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/kyppysimani
[TF]: https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white
[TF-url]: https://www.tensorflow.org/
