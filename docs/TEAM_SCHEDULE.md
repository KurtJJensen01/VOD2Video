# VOD2Video Team Schedule and Workflow Plan

## Goal
Create a work plan for 3 people where everyone is contributing to the app itself, but the work is organized by **dependency order**.

This means:
- some things must be done first
- once they are done, multiple other tasks can begin
- the team should always have parallel work available
- nobody should be stuck waiting if another path is available

---

## Main idea
We are not dividing the project by isolated person ownership.
We are dividing it by **workflow branches**.

Each branch has:
- a dependency
- a start point
- a set of tasks
- a clear handoff to the next branch

All 3 teammates will work on the app, but each person can pick up tasks from different branches as soon as their dependency is satisfied.

---

# Phase 0 — Scope Lock
This must happen first before anything else.

## Must be true before the rest starts
- Dataset is finalized enough to use
- Both CSVs have the same columns
- `vod_id` exists
- `label` exists and means:
  - `1 = highlight`
  - `0 = non-highlight`
- `clip_path` is relative
- Clip folders are organized and working
- We are keeping clip length fixed at 5 seconds
- We are not adding new large features right now

## Output of Phase 0
A stable dataset format and stable project scope.

---

# Phase 1 — Shared Foundation
This is the first real build phase. These tasks should be done immediately because many later tasks depend on them.

## Branch 1A — Dataset Loader — Kurt
### Dependency
Phase 0 complete

### Tasks
- Create code to load both CSV files
- Read all labeled rows
- Keep `vod_id` and `segment_id`
- Build a unique key like `vod_id + segment_id`
- Validate clip paths exist
- Return a single usable in-memory dataset

### Output
A dataset loader that works across both VOD CSVs.

---

## Branch 1B — Project Structure Cleanup - Drew
### Dependency
Phase 0 complete

### Tasks
- Confirm repo folder structure
- Confirm where tools/scripts/models/output files live
- Confirm naming conventions
- Confirm where training artifacts will be saved

### Output
A clean shared structure so later code does not get scattered.

---

## Branch 1C — Train/Val/Test Split Plan — Kurt
### Dependency
Phase 0 complete

### Tasks
- Decide exactly how to split the dataset
- Avoid bad data leakage
- Define whether split is by clip groups, by VOD sections, or by mixed labeled subsets
- Save split logic in code

### Output
A repeatable split strategy.

---

# Once Phase 1 is done, 3 major branches can happen in parallel

---

# Phase 2 — Parallel Branches

## Branch 2A — Feature Pipeline — Kurt
### Dependency
Branch 1A complete

### Tasks
- Decide what features the model will use
- Extract features from each 5-second clip
- Save them into training-ready format
- Make sure labels line up correctly
- Save feature dataset for training

### Suggested first feature set
- audio-based values
- motion-based values
- scene/filler-related values
- any existing useful heuristic signals already available

### Output
Training-ready feature dataset.

### Why this matters
Model training cannot start until this is working.

---

## Branch 2B — Training Framework — Kurt
### Dependency
Branch 1A and 1C complete

### Tasks
- Create model file
- Create training loop
- Create validation loop
- Create metric calculation code
- Create checkpoint saving logic
- Create config/hyperparameter settings

### Important note
This branch can be started before feature extraction is fully done, as long as the expected input format is agreed upon.

### Output
A training framework ready to plug the features into.

---

## Branch 2C — Inference / Demo Framework — Kurt
### Dependency
Branch 1A complete

### Tasks
- Create script to load unseen data
- Create prediction output format
- Create clip ranking flow
- Create way to inspect top predictions
- Create output folder structure for scored clips/examples

### Important note
This does not need the final best model yet. It just needs the expected model input/output contract.

### Output
An inference/demo framework that can later plug in the trained model.

---

# Phase 3 — Second Layer Dependencies

## Branch 3A — First End-to-End Training Run — Kurt
### Dependency
Branch 2A and Branch 2B complete

### Tasks
- Plug extracted features into the training code
- Run the first baseline model
- Check that training, validation, and metrics all work
- Save first results

### Output
First trained model and first metrics.

---

## Branch 3B — Prediction Review Pipeline — Kurt
### Dependency
Branch 2C complete and a first trained model exists

### Tasks
- Run the model on held-out data
- Export prediction scores
- Review top positive predictions
- Review false positives and false negatives
- Decide whether features/model need adjustment

### Output
A working prediction review loop.

---

## Branch 3C — Result Visualization — Kurt
### Dependency
Branch 2B complete

### Tasks
- Build confusion matrix output
- Build metric tables
- Build result plots if needed
- Save figures for report/presentation

### Output
Presentation-ready results visuals.

---

# Phase 4 — Improvement Pass

## Branch 4A — Feature Improvement — Kurt
### Dependency
Branch 3A complete

### Tasks
- Review mistakes
- Adjust features if needed
- Remove bad/noisy features if needed
- Add one useful improvement if needed

### Output
Improved feature set.

---

## Branch 4B — Model Improvement — Kurt
### Dependency
Branch 3A complete

### Tasks
- Tune one or two parameters only
- Try one model improvement only
- Avoid running too many experiments
- Lock best model after comparison

### Output
Final model choice.

---

## Branch 4C — Demo Example Selection — Kurt
### Dependency
Branch 3B complete

### Tasks
- Pick strongest correct predictions
- Pick meaningful mistakes
- Pick clips for presentation/demo
- Prepare side-by-side examples

### Output
A curated set of final demo examples.

---

# Phase 5 — Final Model Lock

## Branch 5A — Final Metrics Package
### Dependency
Branch 4B complete

### Tasks
- Choose final feature setup
- Choose final model configuration
- Choose final checkpoint
- Choose final decision threshold / selection logic
- Save final metrics
- Save confusion matrix
- Save experiment comparison table
- Write final observations

### Output
Final locked model and technical results package.

---

## Branch 5B — Final Demo Example Package
### Dependency
Branch 4C complete

### Tasks
- Organize best true positives
- Organize meaningful false positives
- Organize meaningful false negatives
- Organize borderline examples
- Prepare clip example set for final demo use

### Output
Final example-selection package for the end-user demo.

---

# Phase 6 — New VOD Inference Pipeline

### Dependency
Branch 5A complete

### Tasks
- Take a new unlabeled VOD as input
- Break the VOD into fixed-length clips
- Save generated clips in the expected structure
- Run feature extraction on the new clips
- Run inference on the extracted features
- Score and rank clips by highlight-worthiness
- Save ranked outputs and selected clips

### Output
Working new-VOD pipeline that turns one raw VOD into ranked highlight candidates.

---

# Phase 7 — Highlight Clip Selection

### Dependency
Phase 6 complete

### Tasks
- Select top highlight clips from ranked outputs
- Remove weak or redundant clips if needed
- Keep clips that best represent the final recap
- Save selected clips for final video assembly

### Output
Final selected highlight clip package for video assembly.

---

# Phase 8 — Final Video Assembly

### Dependency
Phase 7 complete

### Tasks
- Organize selected highlight clips
- Put clips in a coherent order
- Add a simple hook / teaser at the beginning
- Merge selected clips into one final condensed video
- Export final video in demo-ready format

### Output
Final generated highlight video package.

---

# Phase 9 — Final Report and Delivery

### Dependency
Branch 5A complete and Phase 8 complete

### Tasks
- Write methodology summary
- Write dataset summary
- Write results summary
- Add visuals
- Add demo examples
- Add final generated video results
- Build slides

### Output
Submission-ready report, final presentation, and final demo package.
