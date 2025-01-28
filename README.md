# **Unsupervised Backdoor Detection and Mitigation for Spiking Neural Networks**  
*Corresponding code for the paper: "Unsupervised Backdoor Detection and Mitigation for Spiking Neural Networks"*

---

## **Description**  
Provide a concise description of the project, its purpose, and its significance.

Example:  
This repository contains the implementation and experimental results for the paper **"Unsupervised Backdoor Detection and Mitigation for Spiking Neural Networks"**, which explores backdoor detection and mitigation defenses against existing attacks in Spiking Neural Networks. The proposed methods, datasets, and evaluation details are included for reproducibility.

---

## **Features**
- Key highlights of the project.
- Briefly list main features, e.g.:
  - **Novel method**: Temporal Membrane Potential Backdoor Detection (TMPBD).
  - **Mitigation mechanism**: Neural Dendrites Suppression Backdoor Mitigation (NDSBM).
  - Comprehensive evaluation on neuromorphic datasets.

---

## **Repository Structure**
```plaintext
├── spikingjelly/               # SNN package 
├── models/                     # Trained models
│   ├── autoencoder/            # Autoencoder models for dynamic attacks
├── data/                       # The data need to manual download following instruction below
├── experiments/                # Experiment results
├── plots/                      # Generated figures and visualizations
├── README.md                   # Project overview
└── requirements.txt            # Required Python packages
```

---

## **Installation**
Provide instructions for setting up the environment and dependencies.

Example:  
1. Clone the repository:

3. Create a conda environment:
    ```bash
    conda create -n snnbd python=3.9 -y
    conda activate snnbd
    ```

4. Install dependencies:
    It is recommended to install cupy and torch manually instead by:
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    conda install -c conda-forge cupy
    ```

5. Install ```spikingjelly``` package:
    Please download from GitHub and put it in the root folder instead of ```pip install spikingjelly``` as advised in [issue](https://github.com/fangwei123456/spikingjelly/issues/401).
    ```bash
    cd TMPBD
    git clone https://github.com/fangwei123456/spikingjelly.git
    ```
    
   
---

## **Usage**
1. Prepare datasets:
    - Place the datasets in the `data/` folder following [this](https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based_en/neuromorphic_datasets.html) instruction.

2. Prepare models:
   Create folder ```models``` and ```models/autoencoder```.
    ```bash
    bash scripts/get_models.sh
    ```
    100 classifier models and 30 autoencoder models should be trained.

3. Backdoor detection experiment:
    ```bash
    bash scripts/detect.sh
    bash scripts/detect_ann.sh
    ```
    Results in ```experiments/detection_results.csv``` and ```experiments/ann_results.csv```.

4. Backdoor mitigation experiment:
    ```bash
    bash scripts/mitigate.sh
    ```
    Results in ```experiments/mitigation.csv```

5. Imbalanced dataset experiment:
    ```bash
    bash scripts/imbalance.sh
    ```
    Eleven models should be trained. The result is shown in ```experiments/detection_results.csv```.

6. Adaptive attack experiment:
    ```bash
    bash scripts/adaptive.sh
    ```
    Six modes should be trained. The result shows in ```experiments/adaptive.csv``` and ```experiments/detection_results.csv```.

---

## **Results**
### Backdoor Detection Accuracy
| #Sample/Class        | DVS128-Gesture       | CIFAR10-DVS         | N-Caltech101       |
|----------------------|----------------------|---------------------|--------------------|
|                      | Clean | Static | Dynamic | Clean | Static | Dynamic | Clean | Static | Dynamic |
| **NC**               |   50   | 100%   | 0%      | 0%    | 100%   | 100%   | 0%    | 100%   | 100%   |
| **ABS**              |   50   | 100%   | 0%      | 0%    | 100%   | 0%     | 0%    | 100%   | 0%     |
| **Neuron Simulation**|    0   | 100%   | 10%     | 50%   | 60%    | 90%    | 100%  | 60%    | 80%    |
| **MMBD**             |    0   | 100%   | 0%      | 100%  | 0%     | 0%     | 100%  | 80%    | 20%    |
| **TMPBD**            |    0   | 90%    | 100%    | 100%  | 80%    | 100%   | 90%   | 90%    | 100%   |

### Attack Label Detection Accuracy
| #Sample/Class        | DVS128-Gesture       | CIFAR10-DVS         | N-Caltech101       |
|----------------------|----------------------|---------------------|--------------------|
|                      | Clean | Static | Dynamic | Clean | Static | Dynamic | Clean | Static | Dynamic |
| **NC**               |   50   | 100%   | 0%      | 0%    | 0%     | 10%    | 10%   | 0%     | 0%     |
| **ABS**              |   50   | 100%   | 0%      | 0%    | 100%   | 0%     | 0%    | 100%   | 0%     |
| **Neuron Simulation**|    0   | 100%   | 0%      | 20%   | 60%    | 60%    | 20%   | 60%    | 50%    |
| **MMBD**             |    0   | 100%   | 0%      | 100%  | 0%     | 0%     | 100%  | 10%    | 50%    |
| **TMPBD**            |    0   | 90%    | 100%    | 100%  | 80%    | 100%   | 90%   | 90%    | 100%   |

### Mitigation Results
|                    | Clean                  | Static                 | Moving                 | Dynamic                |
|--------------------|------------------------|------------------------|------------------------|------------------------|
|                    | **CA**  | **ASR**      | **CA**  | **ASR**      | **CA**  | **ASR**      | **CA**  | **ASR**      |
| **Original**       | 97.65±1.03 | 0.31±0.99  | 98.09±0.99 | 100.00±0.00| 97.21±1.29 | 100.00±0.00| 84.71±12.48| 100.00±0.00|
| **Fine-Tuning**    | 64.56±6.63 | 3.00±4.70  | 56.32±6.97 | 4.38±11.26 | 70.29±13.98| 5.91±12.98 | 88.53±5.54 | 3.28±4.51  |
| **Modified MMBD**  | 73.09±8.38 | 2.90±2.99  | 82.50±4.67 | 46.06±33.33| 73.68±5.11 | 18.34±19.01| 71.76±16.08| 1.40±2.49  |
| **Self-Tuning**    | 7.20±3.13  | 11.44±19.89| 5.88±1.39  | 9.06±20.29 | 7.20±2.81  | 15.72±19.66| 6.47±1.73  | 25.56±20.18|
| **Max Cla.**       | 83.09±3.81 | 2.28±4.30  | 84.41±4.91 | 85.81±20.80| 89.27±4.50 | 75.81±25.78| 88.83±4.00 | 19.38±13.39|
| **Abs. Cla.**      | 84.26±4.75 | 1.34±2.09  | 83.82±7.47 | 84.22±16.31| 87.21±6.05 | 68.13±26.15| 89.12±2.52 | 20.81±14.50|
| **NDSBM**          | 72.50±6.43 | 3.69±10.27 | 72.21±6.56 | 30.41±25.92| 83.38±8.29 | 29.87±19.92| 89.86±3.21 | 8.44±4.91  |
| **TMPBD+NDSBM**    | 97.06±1.55 | 0.31±0.99  | 96.33±3.12 | 19.94±26.48| 95.88±3.00 | 38.12±35.44| 92.06±4.29 | 2.81±3.95  |

## **Citations**
TBA
<!-- ```bibtex
@inproceedings{your_citation,
  title={Your Paper Title},
  author={Your Name and Co-Authors},
  booktitle={Conference or Journal},
  year={2025},
  pages={1234--1245}
}
``` -->

---

## **Acknowledgments**
The implementation of backdoor attacks involved in this research is adopted from work [Sneaky Spikes](https://github.com/GorkaAbad/Sneaky-Spikes).

<!-- 
---

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->

---

## **Contact**
TBA
