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

I designed a **transformer-based temporal model** inspired by PoseFormer, using a **spatio‑temporal transformer architecture** to classify waving at the frame level.

---

## 2.1 Overview of the Pipeline (Expanded)

My pipeline includes both **spatial** and **temporal** modeling using transformers. It operates in two major stages:

---

### **Stage 1 — Spatial Transformer (Per-Frame Encoding)**  
A transformer encoder processes each frame independently, treating the joints inside a frame as input tokens.  
This stage:

- Learns spatial relationships between joints (e.g., wrist ↔ elbow ↔ shoulder)
- Creates a high‑level embedding of each frame
- Reduces noise and stabilizes joint structure

This provides a spatial representation for every frame in the sequence.

---

### **Stage 2 — Temporal Transformer (Sequence Modeling)**  
A second transformer processes the **sequence of encoded frames** to model motion over time.

This stage:

- Learns temporal patterns such as rising arm, oscillating hand, wrist rotations  
- Enables long‑range dependency modeling across all 120 frames  
- Is more expressive and robust than LSTMs for long sequences

Finally, a classification head predicts the waving probability at each frame.

---

## 2.1.1 Supporting Components

Below are the additional modules that improve performance and allow pretrained PoseFormer weights to be used.

---

### **1. Preprocessing & Normalization**  
Coordinates are centered, scaled, and cleaned to reduce inter-person variation and joint noise.

---

### **2. Frame Embedding Layer**  
Each frame is flattened into a 124‑dimensional vector:

```
62 joints × 2 coordinates = 124 features
```

A linear projection maps this into a higher-dimensional embedding (e.g., 256-D), similar to ViT patch embeddings.

---

### **3. Joint Type Embedding**  
To encode *what each joint represents*, I add a learnable embedding:

```
shape = [num_joints (62), embed_dim]
```

This helps the model understand:

- wrists are more important for waving  
- elbows behave differently from shoulders  
- head/torso joints are less discriminative

Joint type embedding improved waving accuracy significantly.

---

### **4. Joint Mapper (62 → 17 Joints)**  
PoseFormer is pretrained on **17-joint COCO format**, while the dataset has **62 joints**.

To leverage these pretrained weights, I designed a learnable **joint mapper**:

- A stack of **1D convolutions along the joint dimension**
- Applied **frame-by-frame**
- Learns to compress 62 joints into a clean 17‑joint representation

```
Input:  62 × 2  
Output: 17 × 2
```

This enables loading **most pretrained PoseFormer parameters**, stabilizing training and reducing noise.

---

### **5. Spatio‑Temporal Transformer**
The core model has two levels:

1. **Spatial Transformer** — models joint relationships inside each frame  
2. **Temporal Transformer** — models motion patterns across the entire sequence  

This structure matches the PoseFormer architecture and is highly effective for gesture and pose dynamics.

---

### **6. Classification Head**
A final linear layer outputs the waving probability for each frame.

---

## 2.2 Why I Chose a Transformer-Based Approach

This was my first time working on action recognition, so I explored the three main classes of approaches:

1. **LSTM-based models**  
2. **Transformer-based models**  
3. **Graph-based models**

---

### **2.2.1 LSTM-Based Models**

I initially considered LSTMs because waving is a sequential action.  
However, LSTMs:

- Struggle with **long sequences** like 120 frames  
- Have limited ability to capture **long-range dependencies**  
- Are sensitive to joint noise  
- Underperform modern transformer methods in the literature

---

### **2.2.2 Graph-Based Methods**

Graph networks treat skeletons as connected graphs. They are powerful, but:

> **I did not have enough time to explore graph-based methods in depth.**

Given their complexity and the unusual 62-joint format, I focused my time on transformers.

---

### **2.2.3 Transformer-Based Models**

After reading **PoseFormer V1 and V2**, I studied their experimental sections.  
Across multiple benchmarks, including 3D pose estimation datasets:

> **Transformers consistently outperform LSTMs**, especially when the sequences are long or the joint inputs are noisy.

This was the **main reason** I shifted from LSTMs to transformers.

Key advantages:

- Self-attention models **long-range motion** effectively  
- More robust to missing or noisy joints  
- Matches naturally with a spatio‑temporal architecture  
- Ability to use **pretrained PoseFormer weights**, improving performance and stability  

Given limited time and strong evidence from PoseFormer experiments, transformers were the most effective choice.

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

- Pretrained PoseFormer clearly improves performance.  
- Data augmentation helps robustness.  
- Joint type embedding gives the **highest waving accuracy**.  
- Joint mapper increases not-waving accuracy but loses fine wrist detail.  
- Best balanced model: **Pretrained + Augmentation + Joint Type Embedding**.

---

## 4. References

1. Zheng et al. *PoseFormer*, ICCV 2021  
2. Zheng et al. *PoseFormerV2*, TPAMI 2023  
3. Vaswani et al. *Attention Is All You Need*, NeurIPS 2017  
4. Donahue et al. *LRCN*, CVPR 2015  
5. Du et al. *Hierarchical RNN*, CVPR 2015  
6. Song et al. *ST-LSTM*, AAAI 2017  

