# VOD2Video

**VOD2Video** is an AI-powered VOD-to-YouTube editor that turns long livestream recordings into condensed, long-form YouTube highlight videos. Instead of generating multiple Shorts, this project focuses on building a full recap video by detecting the best moments, removing filler, creating a teaser intro, and assembling the final output into one cohesive video.

## Overview

Content creators often spend hours reviewing long livestream VODs just to find the best moments for YouTube. VOD2Video is designed to reduce that manual work by using deep learning to identify highlight-worthy clips and automatically generate a recap-style video.

This project is being developed as a deep learning course project and is designed to showcase practical use of:
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks / LSTMs
- Transfer learning
- Optimization methods such as Adam and SGD
- Regularization methods such as dropout, weight decay, and data augmentation

## Project Goal

The main goal of VOD2Video is to build a system that can:
1. Take a full livestream VOD as input
2. Break it into short segments
3. Predict which segments are highlights
4. Remove boring or low-value filler segments
5. Rank the best clips
6. Generate a short teaser intro from the strongest moments
7. Assemble a final long-form YouTube recap video

## Core Features

### Must-Have Features
- Import a full livestream VOD
- Segment the video into fixed-length clips
- Extract frames for model input
- Create and manage labeled training data
- Train a highlight detection model
- Score segments by highlight-worthiness
- Filter out duplicate or overly similar clips
- Remove filler / dead-space where possible
- Select clips for a target recap length
- Generate a teaser intro
- Assemble and export the final recap video

### Nice-to-Have Features
- Adjustable target video length
- Configurable segment size and frame sampling
- Multi-game testing for generalization
- Qualitative review tools
- Simple user interface
- Thumbnail frame suggestions
- Title / description suggestions

## Problem Formulation

The main machine learning task is:

**Binary classification**
- `1 = highlight`
- `0 = non-highlight`

Each sample is a short video segment from a longer VOD. The model outputs a probability score that represents how highlight-worthy the segment is. These scores are then used to rank clips and build the final recap video.

## Model Approach

The planned base model is a **CNN + LSTM architecture with transfer learning**.

### Planned pipeline
1. Sample frames from each video segment
2. Pass frames through a pretrained CNN backbone such as ResNet-18 or ResNet-34
3. Extract frame-level feature embeddings
4. Feed the embeddings into an LSTM to model temporal information
5. Use a fully connected layer to predict highlight probability

### Planned comparisons
- CNN-only baseline vs CNN-LSTM
- Adam vs SGD with momentum
- Transfer learning vs reduced training from scratch
- Different clip lengths: 3s vs 5s vs 10s
- Different regularization setups

## Dataset Plan

The project will primarily use a **custom dataset** built from gaming or livestream VODs.

### Labeling plan
Each VOD will be segmented into fixed-length clips and labeled as:
- **Highlight**
- **Non-highlight**

### Highlight examples
- high-action gameplay
- funny moments
- strong reactions
- wins or close calls
- major events

### Non-highlight examples
- menus
- loading screens
- downtime
- low-energy moments
- repetitive filler

### Dataset split
Recommended split:
- 70% train
- 15% validation
- 15% test

To reduce leakage, splits should be done at the **video level** when possible.

## Planned Experiments

The main experiments for the project are:
1. Compare CNN-only vs CNN-LSTM
2. Compare Adam vs SGD
3. Compare transfer learning vs reduced training from scratch
4. Compare dropout / weight decay settings
5. Compare different clip lengths
6. Evaluate whether the selected clips create a coherent final recap video

## Evaluation

### Quantitative metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### Qualitative metrics
- Are the selected clips actually interesting?
- Does the teaser use the strongest moments?
- Does the final recap feel coherent?
- Does the system remove enough filler to be useful?

## Project Structure

```text
VOD2Video/
├── data/
│   ├── raw_vods/
│   ├── processed_segments/
│   ├── frames/
│   └── labels/
├── notebooks/
├── src/
│   ├── preprocessing/
│   │   ├── segment_video.py
│   │   ├── extract_frames.py
│   │   └── build_dataset.py
│   ├── models/
│   │   ├── cnn_baseline.py
│   │   ├── cnn_lstm.py
│   │   └── transformer_baseline.py
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── metrics.py
│   ├── inference/
│   │   ├── score_vod.py
│   │   ├── rank_segments.py
│   │   ├── teaser_builder.py
│   │   └── assemble_video.py
│   └── utils/
├── outputs/
│   ├── predictions/
│   ├── teaser/
│   └── final_videos/
├── reports/
├── README.md
└── requirements.txt
```

## Team Collaboration

The project is being divided by feature rather than by isolated pipeline stage.

### Feature Group 1: Highlight Intelligence
- VOD segmentation
- frame extraction
- dataset creation
- labeling
- highlight detection model
- filler detection
- evaluation metrics

### Feature Group 2: Video Generation
- segment ranking
- duplicate filtering
- teaser generation
- recap assembly
- export pipeline

### Feature Group 3: Product / Integration
- settings and configuration
- progress logging
- qualitative review
- multi-game testing
- documentation
- integration and demo polish

## Roadmap

### Phase 1
- Finalize project scope
- Collect VODs
- Define labeling rules
- Build segmentation and frame extraction pipeline

### Phase 2
- Build dataset loader
- Implement CNN baseline
- Train first baseline model

### Phase 3
- Implement CNN-LSTM model
- Run core experiments
- Evaluate model performance

### Phase 4
- Build full VOD inference pipeline
- Rank segments
- Generate teaser intro
- Assemble final recap video

### Phase 5
- Polish outputs
- Prepare presentation
- Finalize report and demo

## Current Status

This repository is currently in the planning and setup stage. The proposal, technical spec, and README have been created, and implementation work will begin with dataset creation and preprocessing.

## Future Work

Possible future extensions include:
- audio-aware highlight detection
- thumbnail frame suggestion
- title and description generation
- style modes such as funny, action-heavy, or reaction-heavy
- a lightweight desktop or web interface

## Notes

This project is focused on **long-form YouTube recap generation**, not automatic Shorts generation.

---

## Authors
Project developed by the VOD2Video team as part of a deep learning course project.
