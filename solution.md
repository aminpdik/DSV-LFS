# 1.Problem Summary

The task is to build a classifier that determines, for each frame in a 120-frame sequence of human 2D keypoints, whether the person is waving (True) or not waving (False).

Each sequence contains:



120 frames



62 keypoints per frame (x, y, visibility)



A boolean label “waving” for each frame



Noisy 2D joints (estimated by an unknown pose estimator)



Many negative samples where the person is performing non-wave gestures (pointing, stretching, stop gesture, etc.)



The dataset therefore represents a temporal activity-recognition problem under noisy pose estimation, requiring a model that can:



Understand temporal dynamics of a wave (start → peak → end)



Handle joint noise and missing joints



Generalize to different styles of waving (big arm swings, subtle wrist movement)



The goal is to build a robust classifier with high accuracy on the validation set.

# 2.Method



* 2.1 Overview of My Approach





I implemented a transformer-based spatio-temporal classifier inspired by PoseFormer.

The pipeline consists of:



Preprocessing and normalization



Linear embedding of 62 joints × 2 coordinates



Temporal transformer encoder



Per-frame classification head



This allows the model to capture subtle temporal relations that define a waving gesture.



* 2.2 Why I Chose a Transformer (Detailed Reasoning Based on My Learning Process)



This was my first time working on an action-recognition problem, so I began by surveying the major families of methods used in the literature. Very quickly, I realized that most modern approaches fall into three main categories:



LSTM-based sequential models:



Transformer-based temporal models



Graph-based spatio-temporal networks (GCN/ST-GCN)



Below I explain how I evaluated each and why I ultimately chose transformers for this task.



Step 1 — Initial Intuition: LSTM-Based Models



Since waving is a temporal gesture, I knew immediately that “labeling a single frame” would not be enough.

The model must analyze a sequence of neighboring frames to understand motion.



So my first thought was:



“This is a sequential classification problem. LSTMs or GRUs should work.”



I then read several classical LSTM-based action recognition papers.



These helped me understand how RNNs model temporal dynamics but also highlighted several limitations:



Difficulty capturing long-range dependencies (120 frames in this dataset)



Susceptible to vanishing gradients



Weaker performance on subtle, fine-grained motions



Sensitive to joint noise



While LSTMs were a reasonable baseline, they did not seem ideal for long, noisy sequences.



Step 2 — Evaluating Transformer-Based Models



After reviewing RNNs, I wanted to see if there were more recent alternatives.

I asked ChatGPT for papers using transformers for pose or motion analysis and was directed to:



PoseFormer V1



PoseFormer V2



Other self-attention-based pose estimation approaches



I read the PoseFormer papers and examined their experimental sections. Across multiple benchmarks, transformers consistently outperformed LSTMs for temporal modeling, particularly when the sequence is long or joints are noisy.



Key advantages I identified:



Self-attention naturally captures long-range temporal dependencies



Robustness to missing/noisy joints



Better overall performance in 3D pose reconstruction and action recognition tasks



This strongly suggested that transformers would handle the temporal aspect of waving more effectively than recurrent networks.



Step 3 — Considering Graph-Based Models (GCNs / ST-GCN)



I also investigated graph-based methods such as ST-GCN, which model skeletal joints as graph nodes with edges defined by the human body structure.





Step 4 — Key Practical Advantage: Access to Pretrained PoseFormer



A major deciding factor was that I already had access to a pretrained PoseFormer model for 3D pose estimation.



This was extremely beneficial:



I could reuse learned temporal representations



Fine-tuning became much faster and more stable



Transfer learning improved accuracy compared to training from scratch



The model already understands human motion patterns



In practice, this turned out to be true — using pretrained PoseFormer weights noticeably improved the final performance.



Final Decision



After this exploration, transformers emerged as the most suitable choice because:



They outperform LSTMs on long sequences



They are more robust to noise than GCN-based models



They capture both short- and long-term motion dependencies



I could leverage pretrained PoseFormer weights for better performance



Therefore, I built the solution around a transformer-based temporal architecture, inspired by PoseFormer.





* a
* a
* a
* a
* a
* a
* a
* 

## Implementation details

## References

# Results

## Discussion

