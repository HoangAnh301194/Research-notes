# Methodology: Reading, Analyzing, and Deploying Vision Mamba (ViM)

## Research Focus
Understanding the theoretical mechanics, mathematical formulation, PyTorch code implementation, and edge hardware deployment considerations of Vision Mamba (Vim).

## Step-by-Step Study Pipeline

### Step 1: Mathematical Foundations of Selective SSMs
- Continuous state equation: $h'(t) = \mathbf{A} h(t) + \mathbf{B} x(t)$, $y(t) = \mathbf{C} h(t)$.
- Zero-Order Hold (ZOH) Discretization:
  $$\overline{\mathbf{A}} = \exp(\Delta \mathbf{A}), \quad \overline{\mathbf{B}} = (\Delta \mathbf{A})^{-1}(\exp(\Delta \mathbf{A}) - \mathbf{I}) \cdot \Delta \mathbf{B}$$
- Discrete recurrence:
  $$h_t = \overline{\mathbf{A}} h_{t-1} + \overline{\mathbf{B}} x_t, \quad y_t = \mathbf{C} h_t$$

### Step 2: Bidirectional ViM Block Execution Trace
1. **Input**: Patch tokens $\mathbf{T} \in \mathbb{R}^{B \times M \times D}$.
2. **Linear Projections**: $x = \text{Linear}^x(\text{Norm}(\mathbf{T}))$, $z = \text{Linear}^z(\text{Norm}(\mathbf{T}))$.
3. **Dual-Branch Selective Scan**:
   - For $o \in \{\text{forward}, \text{backward}\}$:
     - $x'_o = \text{SiLU}(\text{Conv1d}_o(x))$
     - Compute data-dependent parameters: $B_o, C_o, \Delta_o = \text{Linear}(x'_o)$
     - Discretize: $\overline{\mathbf{A}}_o = \Delta_o \otimes \mathbf{A}_o$, $\overline{\mathbf{B}}_o = \Delta_o \otimes B_o$
     - Sequential SSM Recurrence to compute $y_o$
   - Gating: $y'_o = y_o \odot \text{SiLU}(z)$
4. **Fusion**: Output $\mathbf{T}_l = \text{Linear}^T(y'_{\text{forward}} + y'_{\text{backward}}) + \mathbf{T}_{l-1}$.

### Step 3: Computational & Memory Complexity Analysis
| Metric | Vision Transformer (DeiT) | Vision Mamba (ViM) |
| :--- | :--- | :--- |
| **Time Complexity** | $O(M^2 D + M D^2)$ | $O(M D N)$ |
| **FLOPs ($E=2D, N=16$)** | $4MD^2 + 2M^2D$ | $8MDN = 128MD$ |
| **GPU Memory Scaling** | $O(M^2)$ (Attention map) | $O(M)$ (Linear scan) |
| **High-Res Batch Inference** | Baseline ($1.0\times$) | $2.8\times$ Faster |
| **GPU Memory Reduction** | Baseline ($0\%$) | $86.8\%$ Saved |

### Step 4: Practical Code Verification & Benchmarking
- Code reference: `https://github.com/hustvl/Vim`
- Verify CUDA kernel fusion (`selective_scan_cuda`) for fast SRAM-based execution.
