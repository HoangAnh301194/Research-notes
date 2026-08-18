Slide 1 — Paper Overview

VideoMamba: State Space Model for Efficient Video Understanding

Proposed by Kunchang Li et al.
Introduces a Mamba-based video backbone.
Targets efficient short-term and long-term video understanding.
Focuses on reducing computational and memory costs for long video sequences.
Slide 2 — Motivation

Challenges in Video Understanding

Videos contain large spatiotemporal redundancy.
Long videos require long-range dependency modeling.
3D CNNs are strong at local modeling but limited for long-range dependencies.
Video Transformers model global relations but suffer from high attention cost.
Slide 3 — Main Idea

From Attention to State Space Models

VideoMamba replaces attention-based sequence modeling with Mamba.
Mamba provides linear complexity with respect to sequence length.
This makes it more suitable for long and high-resolution videos.
The paper reports higher efficiency than TimeSformer in long-video settings.
Slide 4 — Overall Architecture

VideoMamba Framework

Input video: X
v
	​

∈R
3×T×H×W
3D Patch Embedding
Spatial and temporal position embeddings
Bidirectional Mamba blocks
Final classification head

Pipeline:

Video → 3D Patch Embedding → Spatiotemporal Tokens
      → Bidirectional Mamba Blocks → Classification Head

Slide 5 — Video Tokenization

Converting Video into Token Sequences

Patch embedding uses a 1×16×16 3D convolution.
Number of tokens:
L=T×
16
H
	​

×
16
W
	​

Example: 16×224×224 video produces:
16×14×14=3136

tokens.

Slide 6 — Spatiotemporal Scan

Ordering Video Tokens for Mamba

The paper evaluates different scan strategies:
Spatial-First
Temporal-First
Spatiotemporal variants
Spatial-First Bidirectional Scan is selected as the main design.
It processes spatial tokens frame by frame and performs bidirectional sequence modeling.
Slide 7 — Training Strategies

Self-Distillation and Masked Modeling

Larger VideoMamba models may overfit.
Self-distillation uses a smaller trained model as teacher.
Masked modeling improves fine-grained temporal understanding.
The paper studies several masking strategies, including random, tube, row, and attention masking.
Slide 8 — Conclusion

Key Takeaways

VideoMamba adapts Mamba to video understanding.
It models spatiotemporal video tokens with linear complexity.
It performs well on short-term, long-term, and multimodal video tasks.
Remaining limitations include larger-scale models, audio integration, LLM integration, and hour-level video understanding.