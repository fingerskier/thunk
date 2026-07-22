# Model 2: Binary-Weight Auto-Halting Recurrent SLM

## Objective

Build a small recurrent language model that spends compute by repeatedly applying a shared cell, learns when to halt, and ultimately stores its large weight matrices as packed binary values.

The initial goal is not to make every operation binary. The practical target is:

- binary matrix-projection weights;
- recurrent parameter sharing;
- floating-point recurrent state, residual paths, normalization, scales, and halt logic;
- optional binary activations for XNOR-popcount inference;
- a conventional differentiable training representation and a separate packed inference representation.

This separation lets the project establish model quality before introducing specialized packed kernels.

## Representation Decisions

### Canonical packed storage

Use `uint32` or `int32` as the canonical container for 32 binary values. A `float32` has 32 physical bits, but it is not a safe numeric representation of 32 independent weights:

- it has only 24 bits of exact integer precision;
- arbitrary payloads may encode NaNs or infinities;
- runtimes and devices may canonicalize NaNs;
- numeric conversion is not the same operation as a bitcast.

A `float32` may be used as an opaque transport container only when every access uses a verified bitcast and the backend preserves all payload bits. This should be treated as a compatibility experiment, not the primary format.

### Binary convention

Use `{-1, +1}` for learned binary weights and binary activations. Store one sign per bit, for example:

- bit `0` represents `-1`;
- bit `1` represents `+1`.

Record the convention in serialized model metadata. Pad dimensions to the packed word width and carry a valid-bit mask for the final word.

### Training and inference forms

Each binary parameter has two forms:

1. **Shadow parameter:** trainable floating-point tensor `theta`.
2. **Effective parameter:** `b = sign(theta)` used by the forward pass.

Packed integers are derived artifacts for inference and equivalence testing. Optimizers update shadow parameters, never packed words.

## Forward Pass Options

### Arithmetic reference implementation

Represent effective weights and, when applicable, activations as ordinary floating-point `-1/+1` tensors:

```python
binary_weight = binary_sign(shadow_weight)
y = x @ binary_weight.T
```

If `x` is also binary, ordinary multiplication implements sign agreement:

- equal signs produce `+1`;
- unequal signs produce `-1`.

This is the correctness reference. It uses standard framework matrix multiplication and autograd, but does not obtain packed-bit compute savings.

### Packed binary weights with real activations

For real activation `x`, the operation is:

```text
y_j = sum_i x_i * sign(w_ji)
```

Packing the weights reduces storage and memory traffic, but does not turn this into a simple popcount operation. The implementation must either unpack signs during computation or use a specialized kernel that conditionally adds and subtracts activation values.

This is a useful first deployment target because it preserves a high-capacity floating-point state, but expected speedups must be measured rather than assumed.

### Packed binary weights and activations

For `{-1,+1}` weights and activations, a packed dot product is:

```text
matches = popcount(~(weight_bits XOR activation_bits) & valid_mask)
dot = 2 * matches - n_valid
```

XNOR identifies equal signs. Every match contributes `+1`, and every mismatch contributes `-1`.

For multiple words:

```text
acc = 0
for each packed word k:
    matches = popcount(~(w[k] XOR x[k]) & mask[k])
    acc += 2 * matches - valid_bits[k]
```

Accumulate into `int32` or wider, then return to floating point for learned scaling, bias, normalization, residual addition, and halt computation:

```text
packed input
  -> XNOR + popcount
  -> integer accumulation
  -> floating-point scale/bias/norm
  -> residual/nonlinearity
  -> optional sign and repack
```

A learned scale per output row or channel is expected to be important:

```text
y_j = alpha_j * binary_dot(x, w_j) + bias_j
```

For `{0,1}` values, `popcount(weight_bits AND activation_bits)` computes the count of jointly active entries, but `{-1,+1}` is preferred for centered neural representations.

## Backpropagation

### Straight-through estimator

`sign()` and packed bitwise operations have no useful ordinary derivative. Use a straight-through estimator (STE) so the forward pass sees binary values while gradients update floating-point shadow parameters.

A basic STE uses an identity derivative. The initial implementation should use a clipped identity derivative:

```text
b = sign(theta)                         # forward
db/dtheta ~= 1 when abs(theta) <= 1    # backward
             0 otherwise
```

PyTorch-style sketch:

```python
class BinarySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, 1.0, -1.0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * (x.abs() <= 1)
```

The exact zero convention must be fixed and tested. The example maps zero to `+1`.

### Recurrent gradients

Let a shared cell update state as:

```text
h[t+1] = cell(x[t], h[t]; sign(theta))
```

Backpropagation through time proceeds normally through the unrolled recurrence. Because every step reuses `theta`, its gradient is the sum of contributions from all uses:

```text
dL/dtheta = sum_t contribution_from_step_t
```

Start with full BPTT over a small bounded number of recurrent steps. If memory becomes prohibitive, add explicit truncated BPTT as a separately measured option rather than silently detaching state.

Potential recurrent-training stabilizers include:

- pre-normalization;
- a floating-point residual state;
- learned residual or update scales initialized conservatively;
- gradient clipping;
- bounded maximum recurrence;
- monitoring shadow-weight saturation outside the STE gradient window.

### Packed training kernels

Do not require packed bitwise kernels for the first training implementation. Standard framework bitwise operators are generally non-differentiable, and packing/unpacking can erase any training speed advantage. Train using arithmetic `-1/+1` tensors and use the packed path for inference and equivalence tests. Custom autograd or native kernels are a later optimization.

## Recurrent Cell

A practical initial cell is:

```text
u[t]   = BinaryLinear(concat(input_context, h[t]))
h[t+1] = h[t] + update_scale * activation(norm(u[t]))
```

The model should share the cell's binary projections across recurrent steps. Initially retain floating-point:

- token embeddings;
- recurrent state and residual stream;
- normalization parameters and calculations;
- learned binary-layer scales and biases;
- output vocabulary head, unless experiments justify binarizing it;
- halt head and halt probabilities.

This hybrid design isolates the effect of binary weights while preserving state capacity and gradient flow. Binary hidden activations and a fully binary recurrent state are later ablations.

## Auto-Halting

At each recurrent step, compute a floating-point halt probability:

```text
p_halt[t] = sigmoid(halt_head(h[t]))
```

Training should begin with differentiable soft halting or deep supervision at every step. Add a compute penalty:

```text
loss = task_loss + lambda_compute * expected_steps
```

Inference may halt when the probability exceeds a configured threshold. Requirements:

- always enforce a maximum-step guard;
- define a minimum number of steps if immediate halting is degenerate;
- report both task quality and average/percentile step counts;
- test behavior when no step crosses the halt threshold;
- avoid relying on a hard halt decision for the only training gradient.

A straight-through hard halt is an optional experiment after the soft policy is stable.

## Implementation Phases

### Phase 0: Test harness and float recurrent baseline

Implement a small recurrent LM with shared full-precision weights and bounded recurrence.

Acceptance criteria:

- deterministic unit tests run on CPU;
- gradients reach the shared cell from every unrolled step;
- recurrence uses the same parameter objects at every step;
- fixed-step inference works before auto-halting is introduced;
- a tiny dataset can be overfit.

### Phase 1: Arithmetic binary-weight baseline

Add shadow weights, clipped STE binarization, and binary linear projections represented as ordinary `-1/+1` tensors.

Acceptance criteria:

- forward values contain only `-1/+1` effective weights;
- shadow weights receive finite nonzero gradients;
- optimizer steps can flip effective bits;
- save/load preserves shadow weights and effective predictions;
- the tiny overfit test still passes.

### Phase 2: Hybrid recurrent architecture

Use binary weights in the recurrent projections while retaining floating-point state, normalization, residuals, scales, and heads.

Measure against the Phase 0 baseline:

- parameter storage at training and inference time;
- loss and task quality;
- convergence speed;
- recurrent stability by step;
- shadow-weight saturation and bit-flip rate.

### Phase 3: Auto-halting

Add the halt head, bounded soft-halting objective, compute penalty, and hard inference threshold.

Acceptance criteria:

- maximum-step protection cannot be disabled accidentally;
- halting produces finite losses and gradients;
- the model does not collapse to always halting at the first or final step;
- quality-versus-compute curves are recorded across thresholds and penalties.

### Phase 4: Packed reference implementation

Implement pure, easily audited pack/unpack utilities and a CPU XNOR-popcount binary linear reference.

Required property tests:

- pack then unpack is identity for arbitrary dimensions;
- padded tail bits never affect output;
- arithmetic and packed binary dot products agree exactly;
- random matrices agree across multiple packed words;
- zero, all-one, alternating-bit, and odd-sized dimensions are covered;
- serialization preserves packed values exactly.

### Phase 5: Binary activations

Binarize selected hidden projections while retaining a floating residual stream. Compare:

- weight-only binary projections;
- binary projection inputs with float residual state;
- fully binary state as a high-risk ablation.

Do not advance solely on memory reduction. Require acceptable quality, recurrent stability, and measurable inference benefit.

### Phase 6: Optimized inference kernel

Only after equivalence tests pass, evaluate platform-specific vectorized or native kernels. Benchmark complete layers and complete recurrent decoding, not isolated bit operations.

Track:

- packing overhead;
- memory bandwidth;
- popcount throughput;
- integer-to-float conversion and scaling overhead;
- batch-size sensitivity;
- latency per recurrent step and per generated token;
- energy or power when practical.

## Experimental Matrix

Run controlled ablations with matched width, data, optimizer budget, and recurrence limits:

| Variant | Weights | Projection activations | Recurrent state | Compute path |
|---|---|---|---|---|
| A | float | float | float | standard matmul |
| B | binary | float | float | arithmetic reference |
| C | packed binary | float | float | unpack/specialized kernel |
| D | binary | binary | float residual | arithmetic reference |
| E | packed binary | packed binary | float residual | XNOR-popcount |
| F | packed binary | packed binary | binary | XNOR-popcount |

Primary comparisons:

- task quality and perplexity;
- bytes per effective parameter;
- wall-clock training cost;
- inference latency and throughput;
- average recurrent steps;
- stability over long recurrence;
- arithmetic-versus-packed numerical equivalence.

## Risks

### Binary capacity loss

One bit per weight can substantially reduce model quality. Mitigate with learned scales, wider hidden dimensions, float embeddings/heads, residual state, or a small number of higher-precision layers.

### Recurrent instability

Repeated application amplifies quantization errors and may produce fixed points or oscillation. Measure state change, output entropy, and halt probability by step. Preserve normalized float residual paths until stability is demonstrated.

### STE mismatch

The training gradient is a surrogate and may not optimize the discrete model reliably. Track shadow-weight distributions, saturation, and effective bit-flip rates. Compare clipped and identity STEs rather than assuming one is universally best.

### No real speedup

Packed storage does not imply faster execution. Framework unpacking, non-vectorized popcount, small matrices, and repeated packing may dominate runtime. Keep optimization claims conditional on end-to-end benchmarks.

### Float payload portability

Using float payload bits as storage can fail through NaN canonicalization or numeric conversion. Keep integer packed storage authoritative and test any float-bitcast transport on every target backend.

### Premature full binarization

Binarizing embeddings, state, normalization, heads, and halt logic simultaneously makes failures difficult to diagnose. Introduce one quantization boundary at a time.

## Initial Deliverable

The first useful implementation should include:

1. a full-precision shared recurrent baseline;
2. an STE binary linear layer with floating shadow weights;
3. a hybrid float-state/binary-weight recurrent cell;
4. bounded soft auto-halting;
5. arithmetic-versus-packed dot-product tests;
6. a small benchmark reporting quality, storage, and latency.

The default design assumption is therefore **binary packed projection weights with a floating-point recurrent control plane**. A fully packed XNOR-popcount recurrent core remains a measured optimization target, not a prerequisite for validating the model idea.
