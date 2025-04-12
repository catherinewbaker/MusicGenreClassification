# Music Genre Classification Project

## Overview
Project exploring machine learning approaches for automated music genre classification using the FMA dataset and metadata.

## Repository Structure
- `/code/`: Python implementations for genre classification
  - `BasicModels/`: Baseline classification model implementations using sklearn libraries
  - `Processing/`: Data preparation scripts
  - `CombinedFeatures/`: Models using multiple feature types for experiments
  - `SingleFeatures/`: Single feature type experiments
  - `Graphs/`: Visualization generation
- `/fma_dataset/`: Audio data and metadata
- `/Preliminary Results/`: Performance graphs and genre mappings

## Setup
1. Download FMA dataset and verify checksums
2. Run initialPreprocessing.py from `Processing/`
3. Run any file in CombinedFeatures or SingleFeatures to generate results (install dependencies per file)

## Results
- Full experiment results: [Google Sheets](https://docs.google.com/spreadsheets/d/12ju6FKUb24GvlD9G4ej7X8GFPSqCBXqW83ZY1_dFyiI/edit?usp=sharing)
- Technical paper: `PreliminaryResults/MusicGenreClassification.pdf`

## Contributors
- Catherine Baker
- Thomas Davidson
