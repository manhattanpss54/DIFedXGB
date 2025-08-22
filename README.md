# DIFedXGB: A Lightweight Privacy-Preserving XGB Framework for Vertical Federated Learning

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains the official code for the paper **"DIFedXGB: A Lightweight Privacy-Preserving XGB Framework for Vertical Federated Learning"**. DIFedXGB is a novel, lightweight, and privacy-preserving framework for XGBoost in Vertical Federated Learning (VFL), designed to overcome the excessive computational costs of traditional methods that rely on Homomorphic Encryption (HE).

## Overview

Vertical Federated Learning (VFL) enables collaborative model training across different organizations that share the same user base but hold different features. However, existing VFL solutions for XGBoost, like SecureBoost, often use HE for privacy, resulting in high computational overhead that is unsuitable for resource-constrained edge devices.

To address this, we propose **DIFedXGB**. It introduces a **Two-way Data Desensitization** scheme, combining **Pseudonymization** and **Generalization** techniques. This approach provides strong privacy guarantees while drastically reducing computational and communication costs, making it ideal for deployment on IoT edge devices.

### Key Advantages

*   **Lightweight & Efficient**: Eliminates costly HE operations, reducing computational cost by up to 200x compared to SecureBoost.
*   **Strong Privacy**: Prevents leakage of raw data and model parameters, and is secure against collusion attacks from honest-but-curious clients and servers.
*   **Edge-Device Friendly**: Low computational and communication overhead enables deployment on resource-limited IoT terminals.
*   **High Performance**: Achieves model accuracy comparable to centralized XGBoost and outperforms SecureBoost on large datasets.

## Core Technology

DIFedXGB's innovation lies in its two-way data desensitization and secure aggregation:

1.  **Two-way Data Desensitization**:
    *   **Active Client (Holds Labels)**: Applies **Pseudonymization** to the gradients (`gi`, `hi`) calculated by XGBoost, replacing them with fake labels to hide model parameters.
    *   **Passive Client (Holds Features)**: Performs **Generalization** on its local data based on the split points received, creating a binary (0/1) decision matrix that completely conceals the original feature values.
2.  **Secure Aggregation**: The edge server performs matrix multiplication and summation on the desensitized data. Since the data is meaningless without the original mapping, the server cannot learn any sensitive information, enabling secure aggregation.

## Performance Evaluation

We evaluated DIFedXGB on six real-world datasets against **SecureBoost** [15].

### 1. Accuracy (Accuracy)
On large datasets, DIFedXGB achieves accuracy on par with centralized XGBoost and surpasses SecureBoost.

### 2. Computational Cost (Computational Cost)
DIFedXGB's training time is orders of magnitude faster than SecureBoost. For instance, on the HIGGS dataset, DIFedXGB is **208 times faster**.

| Dataset | SecureBoost[15] (s) | DIFedXGB (s) | Speedup |
| :--- | :--- | :--- | :--- |
| **Agaricus** | 4397 ± 506 | 1401 ± 327 | 3.14× |
| **Weather** | 58,161 ± 1939 | 7,461 ± 1297 | 7.80× |
| **SCS** | 78,319 ± 2206 | 6,313 ± 1325 | 12.41× |
| **HeartDisease** | 119,341 ± 2411 | 2314 ± 207 | 51.57× |
| **SUSY** | 286,749 ± 5471 | 1321 ± 132 | 217.07× |
| **HIGGS** | 291,143 ± 5839 | 1397 ± 156 | 208.41× |

### 3. Communication Overhead (Communication Overhead)
DIFedXGB reduces communication overhead by more than half through techniques like bit encoding.

| Dataset | SecureBoost[15] | DIFedXGB | Reduction |
| :--- | :--- | :--- | :--- |
| **Agaricus** | 406.47B | 217.61B | 1.86× |
| **Weather** | 6.51kB | 3.46kB | 1.88× |
| **SCS** | 9.26kB | 4.92kB | 1.88× |
| **HeartDisease** | 19.52kB | 10.67kB | 1.83× |
| **SUSY** | 30.52kB | 16.21kB | 1.87× |
| **HIGGS** | 33.17kB | 17.03kB | 1.93× |

## Quick Start

### Environment Requirements

*   Python 3.11+
*   Dependencies: `xgboost`, `numpy`, `pandas`, `scikit-learn`, `pycryptodome` (for optional HE operations)

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
We thank all the collaborators and institutions that contributed to this research.