# ECLIPSE Project

This repository contains a collection of scripts dedicated to the analysis of stellar light curves using the transit method, with a focus on detecting and characterizing phenomena such as starspots, faculae, and potential signatures of CMEs (Coronal Mass Ejections).

The project is based on the **ECLIPSE** model, originally proposed by Adriana Valio (Valio, 2003), later refactored into Python by Beatriz Duque. In more recent developments, **MCMC (Markov Chain Monte Carlo)** methods and **parallel computing** were incorporated, using C-based scripts to improve performance during model fitting (Pinho, Duque, and Valio).

---

## 📂 Script Structure

Below is an overview of the main scripts and their functionalities:

### 🔧 Modeling and Simulation

**`main_para_programadores.py`**  
Python script for detailed modeling of star–planet systems and generation of synthetic light curves.  
Supports inclusion of:
- starspots  
- faculae  
- moons  
- other phenomena  

Recommended for users with programming experience.

---

**`main-first-steps.ipynb`**  
Introductory notebook with examples of light curve modeling.  
Allows quick simulation of the impact of different components:
- starspots  
- faculae  
- moons  
- CMEs  

Also supports modeling from `.fits` files. Ideal for initial exploration and visualization.

---

**`main-lightcurves-simulations.ipynb`**  
Generates light curves using stellar models derived from solar `.fits` data.  
Enables simulation of:
- active regions  
- stellar events  
- structural variations  

---

**`main-lightcurves-venus-cme.ipynb`**  
Simulation based on the transit of Venus across the Sun.  
Uses real data to reproduce physically consistent scenarios, serving as a strong benchmark for model validation.

---

### 📊 Signal Analysis

**`main-fft-pca-sun-signals.ipynb`**  
Analysis of solar signals using:
- FFT (Fast Fourier Transform)  
- PCA (Principal Component Analysis)  

Focused on characterizing CMEs observed in the Sun.

---

**`main-fft-pca-sun-and-hoststar.ipynb`**  
Extension of the previous analysis, including comparison between:
- solar signals (reference)  
- signals observed in host stars  

Example applied to the star HD189733A.

---

**`main-find-spots.ipynb`**  
Detection of starspots and faculae in real light curves.  
Main workflow:
1. Data download via the `lightkurve` library  
2. Model light curve construction  
3. Subtraction between model and observational data  
4. Identification of residual signatures (spots/faculae)

---

### 🔬 Model Fitting

**`main-mcmc.ipynb`**  
Implements light curve fitting using:
- MCMC algorithm  
- parallelization via C code  

Supports modeling with and without starspots, aiming for best-fit solutions to observational data.

---

### 🌌 Main Project Application

**`main-uv-lightcurves.ipynb`**  
Core notebook of the project.

Integrates:
- ultraviolet light curve analysis  
- star–planet system modeling  
- transit simulation  
- MCMC fitting  

Objective: investigate potential **CME signatures in host stars**.

Main study target:
- Star: HD189733A  
- Planet: HD189733Ab  

---

## 🚀 Final Remarks

This set of scripts enables:
- physically detailed light curve simulations  
- spectral and statistical signal analysis  
- robust fitting of models to real data  

The project can be used for both exploratory purposes and scientific research in exoplanet detection and stellar activity analysis.