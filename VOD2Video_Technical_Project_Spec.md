# VOD2Video Technical Project Spec
**Project:** VOD2Video  
**Purpose:** Shared technical specification for project teammates and AI assistants  
**Version:** 1.0

---

# 1. Project Overview

## Product Vision
VOD2Video is an **AI-powered VOD-to-YouTube editor** that takes a long livestream recording and automatically produces a condensed, long-form YouTube highlight video.

The system should:
- analyze a full livestream VOD
- break it into candidate segments
- score each segment by highlight-worthiness
- remove filler / dead-space segments
- select the best moments
- generate a short teaser intro from the top moments
- assemble the final clips into one long-form recap video

This is **not** a Shorts generator.  
This is a **long-form highlight video generator**.

## Core Problem
Streamers and content creators often have multi-hour VODs and need to manually:
- scrub through footage
- find the good moments
- remove dead space
- build a hook/intro
- edit a final recap video

That process is slow and repetitive. VOD2Video aims to automate the core selection and summarization workflow.

---

# 2. Main Project Goal

Build a deep learning system that can identify highlight-worthy video segments from livestream VODs and use them to generate a coherent long-form YouTube recap video with a teaser intro.

---

# 3. Technical Scope

## In Scope
- video segmentation
- feature extraction from frames
- highlight vs non-highlight prediction
- optional dead-space / filler classification
- ranking of candidate segments
- teaser intro creation
- final video assembly
- model comparison experiments
- evaluation metrics and qualitative review

## Out of Scope
- full production-grade GUI
- cloud deployment
- real-time live editing
- advanced NLP title generation as a main focus
- perfect commercial-level editing polish
- auto thumbnail generation as the primary research task

We can mention future features like titles, descriptions, thumbnails, and timestamps, but the **main technical deliverable** is the highlight detection + long-form summarization pipeline.

---

# 4. Recommended Problem Formulation

## Primary ML Task
**Binary classification**
- class 1 = highlight
- class 0 = non-highlight

Each input sample is a short video segment (for example: 3 to 10 seconds).

The model outputs:
- a probability score that the segment is highlight-worthy

This score is then used for segment ranking and final clip selection.

## Optional Secondary Task
**Filler / dead-space classification**
- menu
- loading screen
- waiting / idle
- low-action gameplay
- highlight

This can either be a second classifier or handled inside the same multi-class setup.

## Why Binary Classification First
Binary highlight classification is easier to:
- label
- train
- evaluate
- explain in the report

It is also easier to convert into a final ranking system for video generation.

---

# 5. Deep Learning Approach

## Base Model
**CNN + LSTM with transfer learning**

### Pipeline
1. Sample frames from each video segment
2. Pass each frame through a pretrained CNN backbone
3. Extract frame-level feature embeddings
4. Feed sequence of embeddings into an LSTM
5. Use final hidden state or temporal pooling
6. Pass to fully connected layer
7. Output highlight probability

## Recommended Backbone Options
- ResNet-18
- ResNet-34
- ResNet-50

Preferred starting point:
- **ResNet-18 or ResNet-34** for simplicity and lower compute cost

## Why This Base Model
This directly showcases course concepts:
- CNNs
- RNN/LSTM
- transfer learning
- fully connected layers
- cross-entropy loss
- backpropagation
- Adam / SGD
- dropout / regularization

---

# 6. Alternative Models for Comparison

At least one comparison model should be tested.

## Option A: CNN-only baseline
- sample a few frames from segment
- extract CNN features
- aggregate by averaging or max pooling
- classify with FC layer

Purpose:
- compare temporal modeling vs no temporal modeling

## Option B: Transformer-based temporal model
- frame embeddings from CNN or ViT
- temporal transformer encoder instead of LSTM

Purpose:
- compare LSTM vs attention-based sequence modeling

## Option C: Vision Transformer (ViT)
- use pretrained ViT as image encoder
- aggregate segment-level information
- classify highlight-worthiness

This is optional if time and compute allow.

---

# 7. End-to-End System Pipeline

## Full Pipeline
1. Input full VOD
2. Preprocess video
3. Split into fixed-length segments
4. Sample frames per segment
5. Extract features
6. Predict highlight score per segment
7. Rank segments
8. Remove overlapping / duplicate-feeling segments
9. Select top clips based on target video length
10. Build teaser intro using top 3 to 5 moments
11. Assemble final recap video
12. Export output video

## Recommended Fixed Segment Lengths to Test
- 3 seconds
- 5 seconds
- 10 seconds

Recommended starting point:
- **5 seconds**

## Frame Sampling per Segment
Recommended:
- 8 frames per segment
or
- 16 frames per segment

Starting point:
- **8 evenly spaced frames per segment**

---

# 8. Dataset Plan

## Main Dataset
Create a **custom VOD highlight dataset** from gaming/livestream videos.

### Data Collection
Use long gameplay or livestream VODs.

Possible sources:
- your own VODs
- public gaming VODs
- gameplay recordings with clear highlight moments

### Labeling Strategy
Split each video into fixed-length segments and label each segment:
- highlight
- non-highlight

### Highlight examples
- high-action combat
- funny moment
- major reaction
- close call
- victory / big fail
- major plot/game event
- high-energy / emotionally strong moment

### Non-highlight examples
- menu navigation
- loading screens
- waiting in queue
- setup / downtime
- repetitive filler
- low-energy idle play
- long pauses / nothing happening

## Suggested Minimum Dataset
For a class project, aim for:
- 10 to 20 full VODs
- several hundred to a few thousand labeled segments

Even a smaller dataset is acceptable if experiments are clearly documented.

## Dataset Split
Recommended:
- 70% train
- 15% validation
- 15% test

Important:
- split by **video**, not by random segment only
- avoid having segments from the same original VOD in both train and test if possible

---

# 9. Data Preprocessing

## Video Segmentation
- divide VOD into fixed windows
- optionally allow overlap between windows

Suggested start:
- non-overlapping 5-second windows

Optional improvement:
- use 50% overlap for better coverage

## Frame Preprocessing
- resize frames to model input size
- normalize based on pretrained backbone requirements
- store frames or extracted tensors efficiently

## Data Augmentation
Possible image augmentations:
- random crop
- horizontal flip
- slight brightness / contrast jitter
- random resized crop

Avoid heavy augmentations that distort gameplay semantics too much.

---

# 10. Training Setup

## Loss Function
- Binary cross-entropy
or
- Cross-entropy loss for two-class classification

## Optimizers to Compare
- Adam
- SGD with momentum

## Regularization
- dropout
- weight decay
- data augmentation

## Suggested Hyperparameter Search Space
- learning rate: 1e-3, 1e-4, 5e-5
- batch size: 4, 8, 16
- dropout: 0.3, 0.5
- epochs: 10 to 30 depending on dataset size

## Early Stopping
Use validation loss or validation F1-score to stop training and save best model.

---

# 11. Evaluation Metrics

## Quantitative Metrics
Use:
- accuracy
- precision
- recall
- F1-score
- confusion matrix

F1-score is especially important if the classes are imbalanced.

## Ranking Quality
Since the end goal is highlight selection, also evaluate:
- how often top-ranked clips are actually good highlights
- precision at top-K if possible

## Qualitative Evaluation
Inspect:
- does the final recap feel coherent?
- does it remove boring content?
- are the chosen clips actually engaging?
- does the teaser use the strongest moments?
- does the final output feel like something a creator would actually use?

---

# 12. Planned Experiments

## Required Core Experiments
1. **CNN-only vs CNN-LSTM**
2. **Transfer learning vs reduced training from scratch**
3. **Adam vs SGD**
4. **With and without dropout / weight decay**
5. **Different clip lengths: 3s vs 5s vs 10s**

## Optional Additional Experiments
6. **LSTM vs Transformer temporal model**
7. **8 frames vs 16 frames per segment**
8. **Overlapping vs non-overlapping segments**

## Main Research Question
Which model and configuration best identify highlight-worthy moments in livestream VODs for long-form video summarization?

---

# 13. Final Output Generation Logic

## Highlight Ranking
After model inference:
- assign each segment a highlight score
- sort segments by score descending

## Diversity / Duplication Filtering
Need to prevent:
- multiple nearly identical adjacent clips
- over-selecting from one short section of the VOD

Basic strategy:
- apply non-maximum suppression style filtering over time
- keep minimum time distance between selected clips

## Final Video Length Target
User selects target output duration:
- 5 minutes
- 8 minutes
- 10 minutes
- 15 minutes

System selects top clips until duration limit is reached.

## Teaser Intro
Use top 3 to 5 highest scoring clips:
- take 1 to 3 seconds from each
- place them at the beginning
- optionally add title card later

## Final Recap Order
Two reasonable options:
1. chronological order after selection
2. strongest-first intro, then mostly chronological recap

Recommended:
- teaser = strongest clips
- main recap = chronological order of selected clips

---

# 14. Suggested Folder Structure

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

---

# 15. Recommended Team Task Breakdown

## Person 1: Data + Preprocessing Lead
Responsibilities:
- collect VODs
- segment videos
- extract frames
- organize labels
- manage train/val/test split

Deliverables:
- cleaned dataset
- label files
- preprocessing scripts

## Person 2: Model + Training Lead
Responsibilities:
- implement CNN baseline
- implement CNN-LSTM model
- train models
- tune hyperparameters
- run experiments

Deliverables:
- trained models
- training scripts
- experiment logs
- evaluation metrics

## Person 3: Inference + Video Assembly Lead
Responsibilities:
- run model on full VOD
- rank segments
- filter duplicates
- build teaser intro
- assemble final recap video

Deliverables:
- inference pipeline
- ranked clip output
- teaser builder
- final video export pipeline

## Person 4: Report + Integration Lead
Responsibilities:
- connect results to project proposal
- write methodology and experiments
- prepare presentation visuals
- collect screenshots and output examples
- maintain repo documentation

Deliverables:
- final report
- presentation slides
- README updates
- ablation/result tables

If team size is smaller, combine these roles.

---

# 16. Deliverables

## Technical Deliverables
- dataset creation pipeline
- labeled segment dataset
- CNN baseline
- CNN-LSTM main model
- training and evaluation scripts
- experiment comparison results
- VOD scoring pipeline
- teaser generation logic
- final recap video export

## Academic Deliverables
- project proposal
- experiment tables
- architecture diagram
- result analysis
- final presentation
- final report

## Demo Deliverables
- one or more example input VODs
- final generated YouTube-style recap video
- teaser intro example
- screenshots of ranked segment outputs

---

# 17. Risks and Mitigation

## Risk 1: Dataset too small
Mitigation:
- use transfer learning
- reduce model size
- keep problem binary
- supplement with public datasets if available

## Risk 2: Labels too subjective
Mitigation:
- define clear highlight labeling criteria
- have more than one person review labels if possible
- use consistent rules

## Risk 3: Final videos feel messy
Mitigation:
- keep chronological ordering in main recap
- enforce clip spacing rules
- manually inspect top selections during debugging

## Risk 4: Model overfits
Mitigation:
- dropout
- weight decay
- augmentation
- early stopping
- smaller backbone

---

# 18. Suggested Development Order

## Phase 1
- finalize problem definition
- collect VODs
- define labels
- build segmentation pipeline

## Phase 2
- build dataset loader
- implement CNN baseline
- train first baseline model

## Phase 3
- implement CNN-LSTM
- train and compare with baseline
- evaluate clip lengths and optimizer choice

## Phase 4
- build inference pipeline on full VOD
- score all segments
- rank and select clips

## Phase 5
- build teaser generator
- assemble final long-form recap video
- prepare demo examples and final report

---

# 19. Short AI Hand-Off Prompt for Teammates

Each teammate can paste the following into their own AI:

## Shared Prompt
You are helping build a class deep learning project called **VOD2Video**.

Project summary:
- VOD2Video is an AI-powered system that converts a long livestream VOD into a condensed long-form YouTube highlight video.
- The main machine learning task is binary classification of short video segments into highlight vs non-highlight.
- The base model should be a **CNN + LSTM with transfer learning**, using a pretrained ResNet backbone for frame-level feature extraction and an LSTM for temporal modeling.
- The end-to-end pipeline is: segment VOD -> sample frames -> predict highlight score -> rank segments -> remove duplicates -> create teaser intro -> assemble final recap video.
- This is not a Shorts generator. It is a long-form recap/highlight editor.

Please help with: [INSERT YOUR TASK HERE]

Constraints:
- Keep the approach aligned with a university deep learning course.
- Prioritize PyTorch implementation.
- Favor clear, modular, explainable code.
- The project should include experiments comparing CNN-only vs CNN-LSTM, Adam vs SGD, transfer learning vs reduced training from scratch, and different clip lengths.
- Evaluation should include accuracy, precision, recall, F1-score, and qualitative review of the final assembled recap video.

---

# 20. Final Project Summary

VOD2Video is a practical deep learning project focused on **video highlight detection and long-form VOD summarization**. The technical core is a **CNN-LSTM model** trained to classify short VOD segments as highlights or non-highlights. The final system uses these predictions to automatically build a YouTube-style recap video with a teaser intro.

This project is strong because it:
- uses class concepts directly
- has real-world usefulness
- supports clear experiments
- produces a compelling final demo

