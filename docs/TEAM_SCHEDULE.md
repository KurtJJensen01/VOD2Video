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

## Branch 3C — Result Visualization
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

## Branch 4A — Feature Improvement
### Dependency
First end-to-end training run complete

### Tasks
- Review mistakes
- Adjust features if needed
- Remove bad/noisy features if needed
- Add one useful improvement if needed

### Output
Improved feature set.

---

## Branch 4B — Model Improvement
### Dependency
First end-to-end training run complete

### Tasks
- Tune one or two parameters only
- Try one model improvement only
- Avoid running too many experiments
- Lock best model after comparison

### Output
Final model choice.

---

## Branch 4C — Demo Example Selection
### Dependency
Prediction review pipeline complete

### Tasks
- Pick strongest correct predictions
- Pick meaningful mistakes
- Pick clips for presentation/demo
- Prepare side-by-side examples

### Output
A curated set of final demo examples.

---

# Phase 5 — Final Integration

## Branch 5A — Final Metrics Package
### Dependency
Model improvement complete

### Tasks
- Save final metrics
- Save confusion matrix
- Save experiment comparison table
- Write final observations

### Output
Final technical results package.

---

## Branch 5B — Final Demo Package
### Dependency
Demo example selection complete

### Tasks
- Organize prediction examples
- Organize best clips
- Prepare demonstration order
- Make the app/demo flow understandable

### Output
Final demo package.

---

## Branch 5C — Final Report/Slides Package
### Dependency
Final metrics package and final demo package complete

### Tasks
- Write methodology summary
- Write dataset summary
- Write results summary
- Add visuals
- Build slides

### Output
Submission-ready report and presentation.

---

# How the 3 teammates should work

## Everyone works on the app, but by branch
At any point, each teammate should pick a live branch that is unblocked.

Example:
- Teammate A works on Branch 2A
- Teammate B works on Branch 2B
- Teammate C works on Branch 2C

Then after that:
- Teammate A may move to Branch 3A
- Teammate B may move to Branch 3C
- Teammate C may move to Branch 3B

This keeps everyone active.

---

# Recommended immediate assignment for tomorrow

## Teammate 1
Start Branch 1A:
- dataset loader
- unique ID handling
- path validation

## Teammate 2
Start Branch 1C and Branch 2B prep:
- split strategy
- model/training skeleton
- metric functions

## Teammate 3
Start Branch 1B and Branch 2C prep:
- repo structure cleanup
- inference output structure
- ranking/prediction review skeleton

Once Branch 1A is done:
- Teammate 1 moves into Branch 2A feature extraction
- Teammate 2 continues Branch 2B training framework
- Teammate 3 continues Branch 2C inference/demo framework

---

# Daily team checkpoints
Every day, ask these 4 questions:
1. What branch was completed today?
2. What new branches are now unblocked?
3. What code/contracts changed that everyone needs to know?
4. What is the next most important dependency to clear?

Keep these check-ins short and focused.

---

# Rules to avoid blocking each other
- Agree on input/output formats before coding in parallel
- Push code in small commits
- Do not wait for perfection before handing off
- When blocked, switch to another open branch
- Do not expand scope unless final results are already working

---

# Minimum success path
If time gets tight, the minimum path is:
- Phase 0
- Phase 1A + 1C
- Phase 2A + 2B + 2C
- Phase 3A + 3B + 3C
- Phase 5A + 5B + 5C

This gives us:
- working dataset pipeline
- trained model
- metrics
- demo outputs
- report/presentation

---

# Final note
The team should always prioritize **clearing dependencies** over polishing optional features.

The project succeeds if we can show:
- custom dataset
- trained neural network
- evaluation metrics
- inference results
- demo-ready examples

Everything else is secondary.
