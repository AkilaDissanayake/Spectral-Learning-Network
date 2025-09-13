# Modulation Classification Using RadioML Dataset

## Overview
This project implements a **machine learning model** for automatic **modulation classification** using the **RadioML dataset**. The model analyzes radio signals and classifies them into different modulation types, which is essential for applications in **cognitive radio, spectrum monitoring, and wireless communication research**.

## Dataset
I used the **RadioML 2016.10a** dataset, which contains  radio signals with various **modulation schemes** (e.g., BPSK, QPSK, 8PSK, QAM16, QAM64, FSK, etc.) under varying **Signal-to-Noise Ratios (SNRs)**.

- Format:Complex-valued I/Q samples  
- Preprocessing: The dataset is normalized and segmented into training and testing sets and reshaped for model input.

## Features
- Signal preprocessing (normalization, reshaping)  
- Training of **machine learning models** (CNNs) wit two input layers  
- Evaluation metrics: **accuracy, confusion matrix**  
- Handles multiple modulation types and SNR level
-Input is I/Q samples and SNR value
-Ouput is modulation scheme
