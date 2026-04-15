# VOD2Video: AI-Powered Long-Form YouTube Video Generation from Livestream VODs

## Goal(s) of the Project
The goal of this project is to develop an AI-powered video editing system, called **VOD2Video**, that automatically converts a long livestream video-on-demand (VOD) into a condensed, publish-ready YouTube highlight video. The system will identify important moments in the VOD, remove uninteresting or low-activity segments, generate a short teaser at the beginning of the final video, and assemble the selected clips into a coherent long-form recap. The project is intended to solve a real content-creation problem: manually reviewing long streams to create engaging YouTube videos is time-consuming and repetitive.

From a machine learning perspective, the main objective is to build a deep learning model that predicts whether a video segment is highlight-worthy or not. A second goal is to compare multiple model and training choices to determine which architecture performs best for highlight detection and video summarization. This fits the final project requirement that the work be non-trivial, application-driven, and centered on developing and training a deep learning model.

## Background of the Project
This project falls primarily under **computer vision** and **video understanding**, with some connection to multimedia summarization. Video summarization is the task of shortening a raw video while preserving its most important or interesting content. Highlight detection is a closely related problem in which a model assigns scores to video segments based on their importance or entertainment value. Prior research has explored supervised, weakly supervised, and unsupervised methods for selecting key video moments, often using temporal modeling to preserve structure across time.

For VOD2Video, the application domain is livestream content creation. Unlike generic video summarization, livestream VOD editing requires identifying exciting moments, filtering out dead space such as loading screens or menus, and preserving enough temporal flow to create a watchable YouTube recap. This makes the project more practical than simply classifying still images and more useful than generating isolated short clips. The final system will act as an intelligent editing assistant that helps creators transform multi-hour stream recordings into a single long-form highlight video.

## Reference Papers
The following papers will guide the project design:

1. **Yao, Mei, and Rui, "Highlight Detection With Pairwise Deep Ranking for First-Person Video Summarization," CVPR 2016.**
2. **Mahasseni, Lam, and Todorovic, "Unsupervised Video Summarization with Adversarial LSTM Networks," CVPR 2017.**
3. **Wei et al., "Learning Pixel-Level Distinctions for Video Highlight Detection," CVPR 2022.**

## Deep Learning Model to Be Used as a Base Model
The base model for this project will be a **CNN + LSTM architecture with transfer learning**. First, video frames sampled from each candidate segment will be passed through a pretrained CNN backbone such as **ResNet-18** or **ResNet-50** to extract visual features. These frame-level embeddings will then be fed into an **LSTM** to model temporal behavior across the segment. Finally, a fully connected output layer will predict a binary label or score indicating whether the segment should be included in the final YouTube video.

This base model is a good fit for the project because it directly uses methods learned in class: convolutional neural networks for visual feature extraction, recurrent neural networks/LSTMs for sequence modeling, transfer learning, fully connected layers, cross-entropy loss, backpropagation, and common optimizers such as Adam and SGD. If time permits, a second model based on a Transformer or Vision Transformer style encoder may also be tested as a comparison model, but the CNN-LSTM system will serve as the primary baseline.

## Experiments Planned
Several experiments are planned in order to evaluate both model performance and practical usability:

1. **CNN-only vs. CNN-LSTM:** Compare a frame-based model with a temporal model to see whether sequence information improves highlight detection.
2. **Transfer learning vs. reduced training from scratch:** Test whether using a pretrained ResNet backbone improves accuracy and convergence.
3. **Adam vs. SGD with momentum:** Compare optimizer choices and their effect on training stability and validation performance.
4. **Regularization experiments:** Evaluate dropout, weight decay, and data augmentation to reduce overfitting.
5. **Different clip lengths:** Compare segment windows such as 3 seconds, 5 seconds, and 10 seconds to determine which captures highlight structure best.
6. **Highlight assembly quality:** After scoring segments, test simple ranking and selection rules to create a final long-form recap video with a teaser at the beginning.

Performance will be measured using classification metrics such as accuracy, precision, recall, and F1-score. In addition, qualitative evaluation will be done by inspecting whether the final assembled video feels coherent, removes filler effectively, and surfaces genuinely strong moments.

## Dataset(s) to Be Used
The project will use a combination of a **custom dataset** and a **public benchmark dataset** if available.

### 1. Custom VOD highlight dataset
The main dataset will be created from recorded gaming or livestream VODs. Each long video will be divided into short fixed-length segments. These segments will then be labeled as **highlight** or **non-highlight** based on manual review. Highlight segments may include high action, strong reactions, funny moments, wins, close calls, or major events. Non-highlight segments may include downtime, menus, waiting, loading screens, or low-energy gameplay. This custom dataset is important because it closely matches the real application of the project.

### 2. Public video highlight/summarization datasets
To support benchmarking and compare against known research settings, the project may also use public video highlight or video summarization datasets that have been used in prior work, such as datasets referenced in research on highlight detection and summarization. Existing research papers in this area evaluate on benchmark video collections with human-created summaries or highlight labels, which makes them useful for validating the proposed approach.

The final dataset description in the full report will include the number of videos used, average video length, number of labeled segments, class balance, preprocessing steps, and train/validation/test splits.

## Expected Outcome
The expected outcome is a working prototype of **VOD2Video** that can take a long livestream recording and automatically generate a condensed YouTube highlight video with a teaser intro. The project should demonstrate that deep learning can reduce the manual effort required to turn raw stream footage into publishable content. In addition to being technically aligned with the course, the project has strong practical value because it targets a real workflow used by streamers and video creators.
