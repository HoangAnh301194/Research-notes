# Research Deliberation: Vision Mamba (ViM) & Visual Sequence Modeling

## Knowledge Consolidation
From `literature-review.md` and the ViM paper (`2401.09417v3.pdf` / `ViM_translating.md`), ViM replaces the quadratic self-attention matrix ($Q K^T$) with dual-branch linear selective scans (forward & backward SSM). For edge computing and high-resolution video/action recognition:
- Attention memory scales as $O(L^2)$ where $L = T \times H \times W / P^2$.
- ViM memory scales as $O(L)$, reducing GPU peak memory by up to $86.8\%$.

## Knowledge Gaps & Contradictions
- **Spatial Order Ambiguity**: 1D raster scan order (left-to-right, top-to-bottom) causes spatial discontinuities across rows.
- **Directional Fusion**: ViM uses simple addition ($y'_{forward} + y'_{backward}$) before final linear projection. Is simple addition optimal compared to concatenated gating or spatial cross-scanning?

## Candidate Hypotheses

### Hypothesis 1: Spatiotemporal Video ViM for Edge Action Recognition
- **H0**: ViM's 1D bidirectional scan suffers severe accuracy degradation on video sequences compared to 3D Convolutions / Space-Time Attention.
- **H1**: Adding temporal scan directions (3D bidirectional SSM or space-time interleaved scanning) enables ViM to achieve state-of-the-art trade-offs between Accuracy and Latency/Memory on edge GPUs.
- **Feasibility**: High.
- **Significance**: Enables real-time action recognition on edge hardware (Jetson / NPU) for high-FPS video streams.

## Selected Direction
- **Chosen Focus**: Deep-dive analysis and implementation roadmap of Vision Mamba (ViM) for visual representation learning and video/action sequence modeling.
- **Success Criteria**: 
  1. Clear step-by-step mathematical breakdown of the ViM block (forward/backward SSM, ZOH discretization, selective gating).
  2. Quantitative computational complexity comparison vs. ViT/DeiT.
  3. Actionable guide for code reading, benchmarking, and edge deployment.
