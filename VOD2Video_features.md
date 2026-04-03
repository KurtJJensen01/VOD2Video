# VOD2Video Core Features

## 1. VOD input and import
This is the starting point of the app.

### What it includes
- upload/load a full livestream VOD
- support common video file formats
- read video metadata like duration, FPS, resolution
- validate that the file can be processed

---

## 2. Video segmentation
The VOD needs to be broken into usable chunks.

### What it includes
- split the VOD into fixed-length segments
- optionally support overlapping windows
- assign each segment start/end timestamps
- store segment metadata

### Example
- 5-second clips across the full VOD

---

## 3. Frame extraction
Each segment needs frames for the model.

### What it includes
- sample frames from each segment
- normalize frame count per segment
- resize/preprocess frames for model input
- save frame paths or tensors

### Example
- 8 evenly spaced frames per 5-second segment

---

## 4. Dataset creation / labeling
This is the training-data feature.

### What it includes
- label clips as highlight or non-highlight
- define clear highlight criteria
- save labels in CSV/JSON format
- create train/validation/test splits

### Optional later
- multiclass labels like funny, intense, filler, reaction

---

## 5. Highlight detection model
This is the main AI feature.

### What it includes
- CNN baseline
- CNN + LSTM main model
- training pipeline
- inference pipeline
- output highlight score/probability for each segment

### Main output
- score from 0 to 1 for how highlight-worthy a segment is

---

## 6. Filler / dead-space detection
This is a practical feature that makes the app more useful.

### What it includes
- detect boring or low-value segments
- identify menus, loading screens, waiting, downtime
- reduce the chance of filler ending up in final output

### This could be
- part of the highlight model
- or a separate rule/model layer

---

## 7. Segment ranking
Once all segments are scored, they need to be ranked.

### What it includes
- sort segments by highlight score
- prioritize best moments
- prepare list of candidates for final video
- optionally combine highlight score with filler penalty

---

## 8. Duplicate / near-duplicate filtering
You do not want several clips that feel the same.

### What it includes
- remove clips that are too close in time
- avoid selecting multiple nearly identical moments
- enforce spacing between chosen highlights

This is important for final pacing.

---

## 9. Target length selection
The final video should match a goal.

### What it includes
- choose target output duration
- stop selection once target length is reached
- adapt number of clips to final video size

### Examples
- 5-minute recap
- 8-minute recap
- 10-minute recap

---

## 10. Teaser intro generation
This is one of the coolest product features.

### What it includes
- take top 3–5 strongest moments
- trim very short pieces from them
- combine them into a quick cold-open teaser
- place teaser at the beginning of the final video

This makes the recap feel like a real YouTube video.

---

## 11. Main recap assembly
This is the actual final-video builder.

### What it includes
- select best clips
- order them intelligently
- preserve enough chronology
- build one long-form recap video
- cut and stitch clips together

### Possible logic
- teaser starts strongest-first
- main recap stays mostly chronological

---

## 12. Clip trimming
Selected clips may still need smarter boundaries.

### What it includes
- trim beginning/end of chosen clips
- avoid awkward starts/stops
- keep the most important moment centered

This can start simple and improve later.

---

## 13. Video export
The final result needs to be usable.

### What it includes
- render/export final recap video
- save teaser version and full version
- control output filename and format

---

## 14. Evaluation metrics
This is the academic side.

### What it includes
- accuracy
- precision
- recall
- F1-score
- confusion matrix
- experiment comparison tables

This feature is for measuring how well the AI works.

---

## 15. Qualitative review tools
Not everything is captured by metrics.

### What it includes
- inspect selected clips
- compare predicted highlights vs labels
- review final recap quality
- note pacing, coherence, and usefulness

This matters a lot for your presentation and report.

---

# Practical / Usability Features

These make it feel more like a real app instead of only a research prototype.

## 16. User settings

### What it includes
- target output length
- segment size
- number of frames sampled
- teaser length
- duplicate filtering strength

---

## 17. Processing progress / logs

### What it includes
- show what stage the pipeline is on
- progress while segmenting, training, or exporting
- error handling/log output

Very useful if you build even a simple interface.

---

## 18. Project configuration

### What it includes
- config file for paths and hyperparameters
- easy model switching
- reusable settings across machines

---

## 19. Example demo outputs

### What it includes
- sample VOD input
- ranked clip output
- teaser output
- final recap output

This is important for showing the project works.

---

# Optional Stretch Features

These are nice, but should come after the core system works.

## 20. Multi-game generalization testing

### What it includes
- train on several games
- test on unseen game(s)
- measure whether the model generalizes

---

## 21. Style modes

### What it includes
- funny mode
- action-heavy mode
- reaction-heavy mode
- story/progression mode

This would change how clips are selected.

---

## 22. Audio-aware highlight detection

### What it includes
- incorporate audio spikes or excitement
- combine visual + audio features
- improve highlight scoring

This is a strong upgrade but adds complexity.

---

## 23. Thumbnail frame suggestions

### What it includes
- find visually strong frames from selected clips
- save top thumbnail candidates

---

## 24. Title / description suggestions

### What it includes
- generate draft YouTube title ideas
- generate simple description text

Useful, but should not be the main focus.

---

## 25. Simple UI

### What it includes
- load a VOD
- choose settings
- run the pipeline
- preview outputs

Only do this if time allows.
