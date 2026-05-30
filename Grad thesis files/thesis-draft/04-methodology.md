<!-- FIRST DRAFT — load-bearing chapter. The human author will refine this carefully. Stay strictly faithful to the paper. -->

# 4. Methodology

This chapter presents DP-FedBN, the record-level differentially private composition of FedBN developed in this thesis. It states the system and threat model, fixes the model architecture, recalls the two preliminaries from Chapter 2 that the construction rests on, gives the FedBN aggregation rule, and then presents the core contribution: a dual-optimizer construction that applies DP-SGD to the parameters that cross the federation boundary while keeping the batch-normalization state local. The chapter closes with the privacy accounting, a complete client-side algorithm, and the implementation pitfalls that a re-implementer would otherwise rediscover the hard way.

The framing is the same measured one used throughout: this work contributes neither a new aggregation algorithm nor a new privacy mechanism. To our knowledge, it is the first record-level DP-SGD composition of FedBN's BatchNorm-locality with explicit treatment of the Opacus engineering obstacle, together with the first systematic medical empirical characterization of that composition. The prior "DP-FedBN" baseline of ADCOL [1] is a party-level Laplace mechanism on aggregated updates (Chapter 3); the construction here is record-level via per-sample gradient clipping.

## 4.1 System and threat model

We assume a federated setting with $K$ clients indexed $k = 1, \dots, K$. Each client owns a local dataset $\mathcal{D}_k$ of size $n_k = |\mathcal{D}_k|$. A global model $f_\theta$ with parameters $\theta$ is trained for $R$ rounds; in each round, every client receives the current global parameters $\theta_t$, performs $E$ epochs of local training on $\mathcal{D}_k$, and returns its updated parameters to the server, which aggregates them and broadcasts the result for the next round.

We adopt the standard federated-learning threat model: an **honest-but-curious server** that faithfully follows the protocol but attempts to infer client data from the parameter updates it observes, together with **possibly malicious peer clients** that may inspect the broadcasts they receive. The adversary's view is therefore exactly the set of objects that cross the federation boundary.

This is what makes the DP-FedBN construction natural. Under DP-FedBN — as under plain FedBN — the only objects that cross the boundary are the non-BatchNorm parameters $\theta^{\overline{\mathrm{BN}}}$ (the convolutional and linear weights and biases). The batch-normalization parameters and running statistics $\theta^{\mathrm{BN}}$ are never aggregated and never broadcast; they remain on the client throughout. Consequently, in this threat model the only objects requiring differential-privacy protection are the $\theta^{\overline{\mathrm{BN}}}$ updates, and those are DP-protected at every local step (Section 4.5). The end-to-end privacy guarantee on $\theta^{\overline{\mathrm{BN}}}$ follows from Opacus's privacy accountant applied per client. Because the batch-normalization state $\theta^{\mathrm{BN}}$ never leaves the client, it requires no DP protection under this model — a parameter the adversary never sees cannot be inferred from what the adversary sees.

Two properties of differential privacy make this argument rigorous rather than merely intuitive. First, **post-processing invariance**: any function computed from a differentially private output is itself differentially private, with no additional privacy cost. The server's size-weighted average of DP-protected client updates is such a function, so the aggregate inherits the clients' guarantee without further machinery. Second, the guarantee is **worst-case over auxiliary information**: it holds regardless of what side information the honest-but-curious server or a malicious peer may hold, which is the property that makes a formal DP claim meaningful against an unknown adversary. Differential privacy is, in this setting, complementary to cryptographic protections such as secure aggregation (Section 3.2): the latter would hide individual updates in transit, while DP bounds what the released model leaks; this thesis is concerned with the latter.

We state the boundary of this model explicitly. If a stronger threat model is required — for instance, one in which the per-client batch-normalization statistics could be inferred through repeated query attacks against a *deployed* personalized model — then additional protections on the deployed model would be needed. We do not address that scenario here and note it as a limitation in Chapter 7.

## 4.2 Model architecture

Every experiment in this thesis uses a single architecture, a BatchNorm-heavy one-dimensional convolutional network (denoted CNN1D) with approximately $4.37 \times 10^5$ parameters. It is deliberately conventional: the contribution lies in the privacy composition and the evaluation, not in the network. The architecture is four convolutional blocks followed by a global average pool and a two-layer classifier:

- **Block 1:** Conv1D($1 \to 64$, kernel 7) → BatchNorm → ReLU → MaxPool(2)
- **Block 2:** Conv1D($64 \to 128$, kernel 5) → BatchNorm → ReLU → MaxPool(2)
- **Block 3:** Conv1D($128 \to 256$, kernel 5) → BatchNorm → ReLU → MaxPool(2)
- **Block 4:** Conv1D($256 \to 256$, kernel 3) → BatchNorm → ReLU
- **Head:** GlobalAveragePool → FC($256 \to 128$) → BatchNorm → ReLU → Dropout($0.5$) → FC($128 \to$ num_classes)

The global average pool makes the network agnostic to input length, so the same architecture serves both the 250-sample MIT-BIH beats and the 1000-sample PTB-XL records. The point relevant to this chapter is that the network contains **five batch-normalization modules** — one after each of the four convolutions and one after the first fully connected layer — and it is the parameters and running buffers of these five modules that the DP-FedBN construction treats specially. The set $\theta^{\mathrm{BN}}$ collects, across all five modules, the learnable affine parameters $(\gamma, \beta)$ and the running buffers $(\mu, \sigma^2)$ (together with the integer batch-count buffer); $\theta^{\overline{\mathrm{BN}}}$ is everything else.

## 4.3 Preliminaries

Two facts established in Chapter 2 are load-bearing here and are recalled without re-derivation. First, **DP-SGD** [5] bounds the influence of any single record by clipping each per-sample gradient $g_i = \nabla_\theta\,\ell(f_\theta(x_i), y_i)$ to norm $C$ and adding Gaussian noise, producing the privatized step of Equation (2.1),
$$
\hat{g} = \frac{1}{|\mathcal{B}|}\left( \sum_{i \in \mathcal{B}} \bar{g}_i + \mathcal{N}\!\big(0, \sigma^2 C^2 I\big) \right),
\qquad \bar{g}_i = g_i \cdot \min\!\Big(1, \tfrac{C}{\lVert g_i \rVert_2}\Big),
$$
whose noise multiplier $\sigma$ an accountant chooses to meet a target $(\varepsilon, \delta)$ guarantee.

Second, the **BatchNorm obstacle**: under batch statistics $\mu_{\mathcal{B}}, \sigma_{\mathcal{B}}^2$, the normalized output for sample $i$ is
$$
\tilde{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon_{\mathrm{BN}}}},
$$
which depends on every other sample $j \neq i$ in the batch through $\mu_{\mathcal{B}}$ and $\sigma_{\mathcal{B}}^2$. The per-sample gradient $g_i$ therefore depends on samples other than $i$, and per-sample clipping no longer bounds the sensitivity of one record. Opacus's `ModuleValidator` detects this and refuses to wrap any model containing a `BatchNorm` module; its standard remedy, `ModuleValidator.fix(model)`, replaces every BatchNorm layer with GroupNorm, which is privacy-safe but removes the per-client BN buffer mechanism that FedBN relies on.

## 4.4 FedBN aggregation

FedBN [9] modifies the aggregation rule, not the local training. Partition the model parameters as $\theta = (\theta^{\mathrm{BN}}, \theta^{\overline{\mathrm{BN}}})$, where $\theta^{\mathrm{BN}}$ collects all batch-normalization scale–shift parameters $(\gamma, \beta)$ and running buffers $(\mu, \sigma^2)$ across all BN layers, and $\theta^{\overline{\mathrm{BN}}}$ is everything else. FedBN aggregates the non-BN parameters by the ordinary size-weighted FedAvg average and leaves the BN parameters untouched on each client:
$$
\theta_{t+1}^{\overline{\mathrm{BN}}} = \sum_{k=1}^{K} \frac{n_k}{\sum_j n_j}\, \theta_t^{(k),\,\overline{\mathrm{BN}}},
\qquad
\theta_{t+1}^{(k),\,\mathrm{BN}} = \theta_t^{(k),\,\mathrm{BN}}.
\tag{4.1}
$$
The BN parameters and buffers stay client-local; the rest is averaged exactly as in FedAvg. Because the BN state never leaves the client, in the federated threat model of Section 4.1 it requires no DP protection.

## 4.5 The record-level DP-SGD construction

The challenge in composing FedBN with DP-SGD is implementational rather than conceptual. Opacus's `ModuleValidator` checks the model architecture *eagerly* when `PrivacyEngine.make_private_with_epsilon` is called, and refuses to proceed if the model contains BatchNorm. The construction must therefore make Opacus accept a BatchNorm-containing model while still privatizing exactly the parameters that cross the boundary, and no others.

We resolve this by **temporarily freezing the BatchNorm parameters** so that the validator treats them as inert, non-trainable modules and proceeds. Concretely, before the privacy engine is attached, each BatchNorm module's `weight` and `bias` have `requires_grad` set to `False`. This is the crux of the construction, and it works because of a detail of how Opacus inspects a model: the `ModuleValidator` iterates only over the *trainable* modules, and `GradSampleModule` registers its per-sample-gradient hooks only on modules that own trainable parameters. A BatchNorm module whose parameters are frozen is therefore invisible to both code paths — the validator does not reject it, and no per-sample-gradient machinery is attached to it. After `make_private_with_epsilon` returns, the BatchNorm parameters are unfrozen so that the plain optimizer can update them locally. The freeze is thus a narrow, surgical maneuver: it lasts only for the duration of the privacy-engine setup call, and its sole purpose is to keep BatchNorm out of Opacus's per-sample-gradient accounting while leaving it fully trainable during the actual training loop.

This is also why **two** optimizers are required rather than one. A single Opacus-wrapped optimizer covering all parameters would attempt to compute per-sample gradients for the BatchNorm parameters — the very thing that is ill-defined and that the validator forbids. By giving the DP-wrapped optimizer $\mathrm{opt}_A$ only the non-BN parameters and giving the plain optimizer $\mathrm{opt}_B$ the BN parameters, the construction routes each parameter group through exactly the update rule it needs: clipped-and-noised for the parameters that cross the boundary, plain for the parameters that stay home. The two optimizers do not interfere, because their parameter sets are disjoint and partition the model.

The result is a **two-optimizer training step**. Each client constructs two optimizers:

- $\mathrm{opt}_A$ — an AdamW [17] optimizer over the non-BN parameters $\theta^{\overline{\mathrm{BN}}}$. This optimizer is the one wrapped by Opacus's `PrivacyEngine`, so it performs DP-SGD per-sample gradient clipping and Gaussian noise addition on the non-BN gradients only.
- $\mathrm{opt}_B$ — a plain AdamW optimizer over the BN parameters $\theta^{\mathrm{BN}}$. This optimizer applies standard, un-noised gradient updates.

At each mini-batch, both optimizers are zeroed, a single forward and backward pass is run, and then both step: $\mathrm{opt}_A$ applies the DP-clipped, noised update to the non-BN parameters, and $\mathrm{opt}_B$ applies the plain update to the BN parameters. The two share the same backward pass; what differs is whether the resulting gradients pass through Opacus's clipping-and-noise machinery before the step. Because $\mathrm{opt}_A$ applies DP-SGD with **per-sample** gradient clipping to the parameters that cross the boundary, the guarantee it produces is a **record-level** $(\varepsilon, \delta)$-DP guarantee — this is the precise sense in which the construction differs from, and the source paper positions as stronger than, the party-level Laplace baseline of ADCOL [1].

On the server side, nothing changes from FedBN: the aggregation rule is Equation (4.1) — average the non-BN parameters across clients weighted by sample count, and do not touch the BN state. The server never sees any client's BN state, so no DP composition is required for it. Differential privacy of an aggregate of DP-protected updates follows by post-processing: once each client's transmitted $\theta^{\overline{\mathrm{BN}}}$ update is DP, any function of those updates — including the size-weighted average the server computes — remains DP.

### 4.5.1 The three differentially private modes

The construction above is best understood next to the two alternatives it is compared against in the experiments. At finite $\varepsilon$, the BatchNorm-heavy CNN1D of Section 4.2 can be combined with DP in exactly three ways, and the implementation realizes all three so that the comparison is like-for-like:

1. **DP-FedAvg (raw BN)** — the naive composition: attach the privacy engine to the unmodified, BatchNorm-containing model and aggregate by FedAvg. This is what a practitioner would try first. It does not run: Opacus's `ModuleValidator` rejects the model at setup time. In the implementation, the resulting exception is caught and re-raised as a typed "DP mode unsupported" error so that the experiment harness records the outcome as a first-class empirical finding (a `REFUSED` cell in the result tables of Chapter 6) rather than crashing. The refusal is the empirical fact that motivates any non-naive composition.

2. **DP-FedAvg+GroupNorm** — the standard Opacus workaround: replace every BatchNorm layer with GroupNorm up front via `ModuleValidator.fix` (Opacus's default substitution replaces a $k$-channel BatchNorm with $\mathrm{GroupNorm}(\gcd(32, k), k)$), then attach the privacy engine to the now-DP-compatible model and aggregate by FedAvg. This runs, and it is the conventional baseline, but — as Chapter 2 explained — it has no per-client BatchNorm state, so it forfeits FedBN's personalization mechanism. When BatchNorm is replaced, both the server's reference model and every client are constructed with GroupNorm so that the parameter key spaces stay aligned for aggregation.

3. **DP-FedBN** — the construction of this chapter: freeze BatchNorm, attach the privacy engine to the non-BN parameters only, then run the two-optimizer step with FedBN aggregation. This is the only one of the three that obtains a record-level DP guarantee while keeping BatchNorm — and therefore FedBN's mechanism — alive.

For the no-DP reference point ($\varepsilon = \infty$) the privacy engine is omitted entirely and training reduces to plain FedBN, FedAvg, or FedAvg+GroupNorm; the GroupNorm topology is still applied in that mode because it is a property of the architecture choice, not of the privacy budget. These $\varepsilon = \infty$ rows isolate the effect of *plumbing in* the DP infrastructure from the effect of the DP *noise* itself, a distinction Chapter 6 uses repeatedly.

## 4.6 Handling batch-normalization statistics under DP and FL

It is worth being precise about *which* parts of the BatchNorm state receive *which* treatment, because the mechanism by which DP affects FedBN, analyzed in Chapter 6, depends on it.

- The **running statistics** $(\mu, \sigma^2)$ are not optimized parameters at all. They are exponential moving averages updated by the forward pass (Section 2.4). They are updated normally during local training and they receive **no DP noise directly**, because they are never the target of either optimizer.
- The **affine parameters** $(\gamma, \beta)$ are optimized, but by the plain optimizer $\mathrm{opt}_B$, not by the DP-wrapped $\mathrm{opt}_A$. They therefore receive standard gradient updates and are **not noised either**.
- The **non-BN parameters** $\theta^{\overline{\mathrm{BN}}}$ are the only objects to which DP noise is applied, via $\mathrm{opt}_A$.

The consequence — important for interpreting the results — is that DP noise enters the BatchNorm behavior only *indirectly*. The noised non-BN updates change the convolutional features, which change the activation statistics that feed the BatchNorm running buffers; so the BN statistics drift under DP even though no noise is ever added to them directly. There is, moreover, no averaging across clients to smooth this drift, because FedBN deliberately keeps the BN state local. Chapter 6 shows that this is the mechanism behind DP-FedBN's partition-dependent behavior.

## 4.7 Privacy accounting

The privacy guarantee is established and tracked per client. When the privacy engine is attached, Opacus calibrates the noise multiplier $\sigma$ to the target $(\varepsilon, \delta)$ over the full course of that client's training — that is, over $R \times E$ local epochs — using its RDP accountant and the sampling rate implied by the batch size and local dataset size (Section 2.5). The guarantee is therefore **end-to-end** over all rounds of the client's participation, not per round. During training, the accountant is queried each round to report the cumulative spent $\varepsilon$, which rises toward the target as rounds accumulate; the achieved $\varepsilon$ at the end of training matches the target to within $0.5\%$ across all finite-$\varepsilon$ runs reported in this thesis. Each client maintains its own accountant over its own data; because the server only ever combines DP-protected updates, the per-client $(\varepsilon, \delta)$ guarantees are not weakened by aggregation.

## 4.8 Algorithm and implementation pitfalls

Algorithm 1 states the full client-side procedure for one round, including the freeze-then-unfreeze setup.

> **Algorithm 1 — DP-FedBN: client-side setup and one round**
>
> **Require:** initial model $f_\theta$ with BN layers; train loader $\mathcal{L}_k$; target privacy budget $(\varepsilon, \delta)$; gradient clipping bound $C$.
>
> **Round setup** (after global broadcast)
> 1. Receive $\theta_t^{\overline{\mathrm{BN}}}$; load into the local model.
> 2. **for** each BN module $m$ **do** $\;m.\texttt{weight}.\texttt{requires\_grad} \leftarrow \text{False};\; m.\texttt{bias}.\texttt{requires\_grad} \leftarrow \text{False}$
> 3. Build $\mathrm{opt}_A \leftarrow \mathrm{AdamW}(\theta^{\overline{\mathrm{BN}}})$.
> 4. $(f_\theta, \mathrm{opt}_A, \mathcal{L}_k) \leftarrow \texttt{PrivacyEngine.make\_private\_with\_epsilon}(\dots)$
> 5. **for** each BN module $m$ **do** $\;m.\texttt{weight}.\texttt{requires\_grad} \leftarrow \text{True};\; m.\texttt{bias}.\texttt{requires\_grad} \leftarrow \text{True}$
> 6. Build $\mathrm{opt}_B \leftarrow \mathrm{AdamW}(\theta^{\mathrm{BN}})$ (plain, no DP).
>
> **Local training** (each minibatch $\mathcal{B}$)
> 7. $\mathrm{opt}_A.\texttt{zero\_grad}();\;\mathrm{opt}_B.\texttt{zero\_grad}()$
> 8. $\hat{y} \leftarrow f_\theta(\mathcal{B}.x);\;\; \ell \leftarrow \mathrm{loss}(\hat{y}, \mathcal{B}.y)$
> 9. $\ell.\texttt{backward}()$
> 10. $\mathrm{opt}_A.\texttt{step}()$ &nbsp;&nbsp;▷ DP-clipped + noised, on non-BN
> 11. $\mathrm{opt}_B.\texttt{step}()$ &nbsp;&nbsp;▷ plain, on BN
>
> **Round end**
> 12. Send $\theta^{\overline{\mathrm{BN}}}$ to the server.
> 13. BN parameters and running statistics remain local.

Two implementation pitfalls bit us during development and would bite a re-implementer the same way; both are non-obvious and are documented here because they are part of what makes the construction usable in practice.

**State-dict key prefixing under `GradSampleModule`.** When Opacus wraps a model, the underlying `GradSampleModule` prepends `_module.` to every parameter and buffer key in the model's `state_dict()`. Any code path that snapshots or restores client model state — and FedAvg/FedBN parameter exchange is exactly such a path — must use the same unwrapping convention consistently, or the keys silently fail to match across the wrapped and unwrapped views. We standardized on a single accessor that returns the inner, unwrapped module and route every `state_dict` interaction through it, so that the aggregated parameter key space is always the plain one (`conv1.weight`, not `_module.conv1.weight`).

**ExpandedWeights incompatibility.** Opacus offers an alternative per-sample-gradient backend, ExpandedWeights (`grad_sample_mode='ew'`), which is faster than the default hooks-based backend. We found it incompatible with both the `BatchMemoryManager`'s logical-batch chunking and the `optimizer.zero_grad(set_to_none=True)` pattern: combining them crashes on the second optimizer step because ExpandedWeights accumulates gradients across calls without clearing them. We reverted to the default hooks-based backend, which — paired with a physical batch size of 96 and four data-loader workers — ran about six times faster than our initial configuration. These choices are recorded so that the privacy guarantee is reproducible exactly, rather than only in spirit.
