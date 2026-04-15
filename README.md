# VOD2Video

**VOD2Video** is an AI-powered VOD-to-YouTube editor that turns long livestream recordings into condensed, long-form YouTube highlight videos.

The project focuses on building a system that can:
- take a full livestream VOD as input
- break it into short segments
- predict which segments are highlights
- rank the best moments
- generate a short teaser intro
- assemble a final long-form recap video

This is a deep learning course project centered on **highlight detection and long-form VOD summarization**.

---

## Current Project Direction

The current project plan is:
- build a **custom labeled dataset** from livestream VODs
- train a model to classify clips as **highlight** or **non-highlight**
- evaluate the model with metrics such as accuracy, precision, recall, and F1-score
- use predictions to support long-form recap generation

The official academic plan is documented in `VOD2Video_Project_Proposal.md`.

---

## Repository Documents

### `README.md`
High-level overview of the project, repository purpose, and document guide.

### `TEAM_SCHEDULE.md`
Current team execution plan.

Use this for:
- dependency order
- parallel work branches
- daily workflow
- figuring out what can start next without blocking teammates

### `VOD2Video_Project_Proposal.md`
Official project proposal.

Use this as the main source of truth for:
- project goal
- academic framing
- planned model direction
- experiments
- dataset plan
- expected outcome

### `VOD2Video_Technical_Project_Spec.md`
Technical handoff/spec document for teammates and their LLM assistants.

Its purpose is to give an AI helper enough context to understand:
- what VOD2Video is
- what the team is trying to build
- the overall system goal
- the planned model direction
- the expected pipeline and technical scope

This file is mainly for teammate support and implementation context.

### `VOD2Video_features.md`
Feature reference list.

This is **not** the main scheduling/workflow document anymore.
It is now mainly a reference list of possible app features and ideas.

### `tools/build_labeling_dataset.py`
Dataset-building utility script.

Use this to:
- generate 5-second clips from a VOD
- create `labels.csv`
- prepare reviewable clip sets for manual labeling

---

## Current Dataset Setup

The current dataset structure is based on:
- **2 VODs**
- **2 CSV label files**
- **2 clip folders**

Each CSV contains labeled clip metadata for one VOD.
The dataset uses:
- `vod_id`
- `segment_id`
- `label`
- relative `clip_path`

Labels are:
- `1 = highlight`
- `0 = non-highlight`

---

## Core Problem Formulation

The main machine learning task is **binary classification**.

Each sample is a short video segment.
The model predicts whether that segment is:
- `1 = highlight`
- `0 = non-highlight`

The output score can then be used to rank clips and support final recap generation.

---

## Main Workflow

1. Build labeled dataset from VOD clips
2. Load and combine labeled clip data
3. Extract training features / inputs
4. Train highlight detection model
5. Evaluate model performance
6. Run inference on unseen clips or VOD segments
7. Use predictions to support long-form highlight video generation

---

## Generated Artifacts

Some files under `artifacts/` are generated outputs and do not need to be committed.

To regenerate split manifests locally, run:

```bash
python tools/test_dataset_split.py --write-dir artifacts/splits/branch_1c
```

--- 

## Notes

- This project is focused on **long-form recap generation**, not Shorts generation.
- `TEAM_SCHEDULE.md` is the active workflow guide for the team.
- `VOD2Video_Project_Proposal.md` is the main academic plan.
- `VOD2Video_Technical_Project_Spec.md` is the LLM/teammate handoff doc.
- `VOD2Video_features.md` is now mainly a feature reference file rather than the main work-division plan.
