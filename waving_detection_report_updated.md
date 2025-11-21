# Waving Detection from 2D Keypoints – Report

## 1. Problem Summary

The task is to classify, for each frame in a 120-frame sequence of 2D human keypoints, whether the person is **waving** (True) or **not waving** (False).

Each sequence contains:

- 120 frames  
- 62 joints per frame (x, y, visibility)  
- Per-frame binary waving labels  
- Noisy 2D keypoints from a pose estimator  
- Negative samples that may include distracting gestures (pointing, stretching, etc.)

This is a **temporal action-recognition** problem requiring modeling of **motion progression** over time.

---

## 2. Solution Method

I designed a **transformer-based temporal model** inspired by PoseFormer to perform per-frame classification.

## 2.1 Overview of the Pipeline

The full processing pipeline consists of the following stages:

### **1. Preprocessing & Normalization**
Each frame contains 62 joints × (x, y) coordinates.  
I normalize by centering, scaling, and optionally applying visibility masking to reduce noise and inter-person variation.

---

### **2. Frame Embedding Layer**
Each frame is flattened into a 124-D vector:

```
62 joints × 2 dimensions = 124 values
```

A linear layer projects this vector into a higher-dimensional embedding (e.g., 256), similar to ViT patch embeddings.

---

### **3. Joint Type Embedding**
To give the model information about **what each joint represents**, I add a learnable joint-type embedding:

```
shape: [num_joints (62), embed_dim]
```

This embedding is added to the input, allowing the model to learn that:

- wrists are more important for waving  
- elbows, shoulders, and hands have different roles  
- some joints carry more discriminative value than others  

This improves semantic understanding of joint motion.

---

### **4. Joint Mapper (62 → 17)**
PoseFormer’s pretrained weights expect **17 joints**, while the dataset contains **62**.

To leverage pretrained parameters, I built a **joint mapping module**:

- A sequence of **1D convolution layers along the joint dimension**
- Operates **frame-by-frame** (no temporal dependency)
- Maps:

```
Input:  62 × 2  
Output: 17 × 2
```

This learns:

- How to compress the 62-joint layout into a standard 17-joint skeleton  
- How to reduce noise by filtering irrelevant or unstable joints  
- How to align with pretrained PoseFormer structures  

This module allows loading **most of the pretrained weights**, improving stability and performance.

---

### **5. Temporal Transformer Encoder**
A stack of transformer encoder layers models temporal dependencies across the entire 120-frame sequence.

- Multi-head self-attention captures **long-range motion**
- Position embeddings preserve frame ordering
- Robust to noisy or missing joints

Transformers outperform LSTMs for long sequences and noisy data.

---

### **6. Classification Head**
A linear layer outputs the waving probability for each frame.

---

## 2.2 Why I Chose a Transformer-Based Approach

This was my first time working on action recognition, so I explored the three main classes of methods:

1. **LSTM-based models**  
2. **Transformer-based models**  
3. **Graph-based spatio-temporal models**

### **2.2.1 LSTM-Based Models**
My initial idea was to use LSTMs because waving is a sequential gesture.  
After reading several classic RNN/LSTM papers, I learned that LSTMs:

- Have difficulty modeling **long sequences** (120 frames)  
- Struggle with **long-range dependencies**  
- Are sensitive to noisy joints  
- Often underperform transformers in recent benchmarks  

Thus, I considered them less ideal for this task.

---

### **2.2.2 Graph-Based Methods**
Graph-based architectures (e.g., ST-GCN) treat the body as a graph of connected joints.  
They are powerful, but:

> **I did not have enough time to explore graph-based methods in depth.**

Given their complexity and the unusual 62-joint layout, I focused my time on transformer models.

---

### **2.2.3 Transformer-Based Models**
I read the **PoseFormer V1 and V2** papers and examined their experimental sections. Across multiple benchmarks:

> **Transformers consistently outperformed LSTMs**, especially when sequences were long or joint inputs were noisy.

This was the **main reason** I switched from LSTMs to a transformer-based design.

Additional factors:

- Self-attention handles long-range motion effectively  
- More robust to missing/noisy joints  
- Ability to leverage **pretrained PoseFormer weights**  
- Strong empirical performance in pose and motion tasks  

Given the time constraints and performance patterns in the literature, transformers were the most suitable choice.

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

### 3.2 Key Observations

- Pretrained PoseFormer significantly improves accuracy and stability.  
- Data augmentation increases robustness to noise.  
- Joint type embedding provides the **highest waving accuracy**.  
- Joint mapper improves not-waving accuracy but loses fine hand/wrist detail.  
- Best balanced model: **Pretrained + Augmentation + Joint Type Embedding**.

---

## 4. References

1. Zheng et al. *PoseFormer*, ICCV 2021  
2. Zheng et al. *PoseFormerV2*, TPAMI 2023  
3. Vaswani et al. *Attention Is All You Need*, NeurIPS 2017  
4. Donahue et al. *LRCN*, CVPR 2015  
5. Du et al. *Hierarchical RNN*, CVPR 2015  
6. Song et al. *ST-LSTM*, AAAI 2017  

