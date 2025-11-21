# Waving Detection from 2D Keypoints – Report

## 1. Problem Summary

The task is to classify, for each frame in a 120-frame sequence of 2D human keypoints, whether the person is **waving** (True) or **not waving** (False).

Each sequence contains:

- 120 frames  
- 62 joints per frame (x, y, visibility)  
- Per-frame binary waving labels  
- Noisy 2D keypoints from a pose estimator  
- Negative samples that may include distracting gestures (pointing, stretching, etc.)

This is a **temporal action-recognition** problem requiring modeling of **motion progression** over time, not individual frames.

---

## 2. Solution Method

I designed a **transformer-based temporal model** inspired by PoseFormer to perform per-frame classification.

### 2.1 Overview of the Pipeline

1. **Preprocessing & normalization**  
2. **Frame embedding** (linear layer)  
3. Optional **joint mapping layer (62→17)**  
4. Optional **joint type embedding**  
5. **Temporal transformer encoder**  
6. **Classification head** predicting waving/not-waving per frame

---

## 2.2 Why I Chose a Transformer-Based Approach

This was my first time working on action recognition, so I began by exploring the main families of methods:

1. LSTM-based models  
2. Transformer-based models  
3. Graph-based spatio-temporal models  

### 2.2.1 LSTM-Based Models

I initially considered LSTMs since waving is a sequential motion.  
However, after reading several classical papers, I found that LSTMs:

- Struggle with long sequences (120 frames)  
- Do not capture long-range dependencies well  
- Are sensitive to noisy joints  

Thus, LSTMs were not ideal.

### 2.2.2 Graph-Based Models

Graph models (e.g., ST-GCN) work well with clean, stable skeletons.  
But they are less suitable here due to:

- Noisy/missing joints  
- Uncommon 62-joint format  
- Limited dataset size  
- No readily available pretrained GCNs for this layout  

### 2.2.3 Transformer-Based Models

After reading **PoseFormer V1 and V2**, I observed that:

> Transformers consistently outperform LSTMs on long, noisy sequences.

This was the **main reason** I switched to transformers.

Additional reasons:

- Self-attention models long-range motion naturally  
- More robust to noise  
- Access to pretrained PoseFormer weights provided a strong initialization  

---

## 3. Experimental Results

### 3.1 Summary Table

| Model Variant | Loss | Accuracy | Waving Acc | Not-Waving Acc |
|---------------|--------|----------|------------|----------------|
| Original PoseFormer | 0.6221 | 81.42% | 57.46% | 88.59% |
| + Pretrained + Aug | 0.7906 | **82.78%** | 60.49% | **89.45%** |
| + Pretrained + Aug + Mapper | **0.6037** | 81.32% | 46.97% | **91.61%** |
| + Pretrained + Aug + Joint Type | **0.5780** | 82.18% | **63.43%** | 87.80% |
| + Pretrained Only | 0.9141 | 82.53% | 60.70% | 89.07% |

---

## 3.2 Key Observations

- Pretrained PoseFormer improves performance across the board.  
- Data augmentation increases robustness.  
- Joint type embedding yields the **highest waving accuracy**.  
- Joint mapper improves not-waving accuracy but harms waving detection.  
- Best balanced model: **Pretrained + Aug + Joint Type Embedding**.

---

## 4. References

1. Zheng et al. *PoseFormer*, ICCV 2021  
2. Zheng et al. *PoseFormerV2*, TPAMI 2023  
3. Vaswani et al. *Attention Is All You Need*, NeurIPS 2017  
4. Donahue et al. *LRCN*, CVPR 2015  
5. Du et al. *Hierarchical RNN*, CVPR 2015  
6. Song et al. *ST-LSTM*, AAAI 2017  
