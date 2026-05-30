# 2. Fundamental Concepts

This chapter develops the background needed for the rest of the thesis. It is deliberately self-contained: each concept that later chapters rely on is introduced here once, with the equations and definitions that the methodology and results will reference. The chapter proceeds from the application domain inward — first the electrocardiogram and the clinical classification task, then the deep-learning models used to read it, then the federated-learning setting and its characteristic non-IID difficulty, then batch normalization and the FedBN remedy, then differential privacy and DP-SGD, and finally the precise mechanical conflict between batch normalization and DP-SGD that is the technical heart of this work.

## 2.1 ECG signals and arrhythmia classification

The electrocardiogram is a recording of the heart's electrical activity, measured as a time-varying voltage between electrodes placed on the body surface. Each cardiac cycle produces a stereotyped waveform whose components are named by convention: the **P wave** (atrial depolarization), the **QRS complex** (ventricular depolarization, dominated by the sharp **R peak**), and the **T wave** (ventricular repolarization). The R peak is the most prominent landmark and is the standard reference point for segmenting a continuous recording into individual heartbeats.

A clinical ECG is typically recorded from multiple **leads** — different electrode configurations that view the heart's electrical axis from different angles. The standard diagnostic ECG uses twelve leads; research and monitoring settings often use fewer. Different leads emphasize different pathologies, and the choice of which lead(s) to analyze is a substantive modeling decision, as Chapter 5 discusses for the single-lead PTB-XL task.

Two classification granularities appear in this thesis, and they are genuinely different problems:

- **Beat-level classification** asks, for each individual heartbeat, which rhythm class it belongs to. The reference standard is the **ANSI/AAMI EC57** recommendation [19], which groups the dozens of fine-grained annotation symbols in databases such as MIT-BIH into five clinically meaningful super-classes: **N** (normal and bundle-branch-block beats), **S** (supraventricular ectopic beats), **V** (ventricular ectopic beats), **F** (fusion beats), and **Q** (unknown or paced beats). This grouping is what makes beat classifiers comparable across studies [18], [20], [21].

- **Record-level classification** asks, for an entire ten-second recording, which diagnostic category it represents — for example, normal versus abnormal, or a finer split into diagnostic super-classes such as myocardial infarction, ST/T change, conduction disturbance, and hypertrophy, as defined in the PTB-XL dataset [22].

A defining and unavoidable property of both tasks is **severe class imbalance**. In a representative beat database the normal class dominates — roughly four out of five beats are normal — while clinically critical classes such as fusion beats are vanishingly rare. A classifier that predicts "normal" for everything would achieve high accuracy and be clinically useless. This single fact dictates much of the evaluation methodology in Section 2.2 and Chapter 5: accuracy is a misleading score here, and a class-balanced metric is mandatory.

## 2.2 Deep learning for ECG

An ECG beat or record is, computationally, a one-dimensional time series — a sequence of voltage samples. The natural deep-learning architecture for such data is the **one-dimensional convolutional neural network (1-D CNN)**, which has become a workhorse for ECG classification and, on narrow tasks, has reached expert-level performance [25], [26], [27].

A 1-D convolutional layer slides a bank of learnable filters along the time axis. For an input feature map $x \in \mathbb{R}^{C_{\text{in}} \times L}$ with $C_{\text{in}}$ channels and length $L$, a convolution with kernel $w \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k}$ produces, for output channel $o$ and position $t$,

$$
(x * w)_{o,t} = b_o + \sum_{c=1}^{C_{\text{in}}} \sum_{j=1}^{k} w_{o,c,j}\, x_{c,\, t+j-1},
$$

where $k$ is the kernel size and $b_o$ a bias. Stacking such layers, interleaved with pointwise nonlinearities such as the rectified linear unit $\mathrm{ReLU}(z) = \max(0, z)$ and with **pooling** layers that downsample along time, lets the network build a hierarchy of temporal features: early layers respond to local waveform shapes (the slope of an R peak, the width of a QRS complex), later layers to longer-range morphology. A **global average pooling** layer at the end collapses the time axis to a fixed-length vector regardless of input length, which is convenient when beats and records of different durations must be handled by one architecture. One or more fully connected layers then map this vector to class scores, and a softmax converts the scores to class probabilities,

$$
p_c = \frac{\exp(z_c)}{\sum_{c'} \exp(z_{c'})},
$$

with training driven by the cross-entropy loss $\ell(p, y) = -\log p_y$ for true label $y$.

Because of the class imbalance noted in Section 2.1, model quality in this thesis is reported with the **macro-averaged F1 score** rather than accuracy. For a single class $c$, with $\mathrm{TP}_c$, $\mathrm{FP}_c$, and $\mathrm{FN}_c$ counting true positives, false positives, and false negatives, precision and recall are

$$
\mathrm{precision}_c = \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FP}_c}, \qquad
\mathrm{recall}_c = \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FN}_c},
$$

their harmonic mean is the per-class F1 score,

$$
\mathrm{F1}_c = \frac{2\,\mathrm{precision}_c \cdot \mathrm{recall}_c}{\mathrm{precision}_c + \mathrm{recall}_c},
$$

and the **macro F1** averages the per-class scores with equal weight,

$$
\text{macro-F1} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \mathrm{F1}_c .
$$

The equal weighting is the crucial property: a rare but clinically important class contributes as much to macro F1 as the dominant normal class, so a model cannot score well by ignoring minorities. Class imbalance is further mitigated during training by **inverse-frequency class weighting** of the cross-entropy loss, which up-weights the gradient contribution of under-represented classes; the exact weighting used here is described in Chapter 5.

## 2.3 Federated learning and the non-IID problem

Federated learning [2], [16] trains a shared model across $K$ clients without centralizing their data. Client $k$ holds a local dataset $\mathcal{D}_k$ of size $n_k = |\mathcal{D}_k|$, and the total dataset size is $n = \sum_{k=1}^{K} n_k$. The learning objective is the size-weighted average of the clients' local objectives,

$$
\min_{\theta}\; F(\theta) = \sum_{k=1}^{K} \frac{n_k}{n}\, F_k(\theta), \qquad
F_k(\theta) = \frac{1}{n_k} \sum_{(x,y) \in \mathcal{D}_k} \ell\big(f_\theta(x), y\big),
$$

which is exactly the objective that pooled central training would minimize, but it must be optimized without ever moving the data.

**Federated Averaging (FedAvg)** [2] is the canonical algorithm. Training proceeds in **rounds**. In round $t$, the server broadcasts the current global parameters $\theta_t$; each client initializes its local model to $\theta_t$, performs $E$ epochs of local stochastic gradient descent on $\mathcal{D}_k$ to obtain updated parameters $\theta_t^{(k)}$, and returns them to the server; the server aggregates by a size-weighted average,

$$
\theta_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n}\, \theta_t^{(k)},
$$

and the next round begins. Performing several local epochs per round, rather than communicating after every gradient step, is what makes FedAvg communication-efficient.

FedAvg is provably well-behaved when the client data is **independent and identically distributed (IID)** — when every client's data looks like a random sample from one common distribution. Real federated data almost never satisfies this. The general term for the violation is **non-IID** or **statistically heterogeneous** data, and it comes in several distinguishable forms that this thesis treats separately:

- **Label (prior) skew:** clients have different class proportions. A specialty clinic may see almost exclusively one or two conditions; a referral center sees a different mix. In the extreme, a client holds only a subset of the label space.
- **Covariate (feature) skew:** the inputs themselves are distributed differently across clients — different acquisition devices, electrode placements, or patient demographics shift the feature distribution even for the same label.
- **Quantity skew:** clients simply hold very different *amounts* of data, independent of class proportions.

Under heterogeneity, the local optima $\theta_t^{(k)}$ that the clients reach in a round can point in conflicting directions; their average can be a poor compromise that fits no client well. This phenomenon, **client drift**, makes FedAvg converge more slowly and to a worse solution than centralized training, and the effect worsens as heterogeneity increases. Two influential responses frame the rest of this thesis.

**FedProx** [11] keeps FedAvg's aggregation but modifies the local objective, adding a proximal term that anchors each client's local solution to the broadcast global model:

$$
\min_{\theta}\; F_k(\theta) + \frac{\mu}{2}\,\big\lVert \theta - \theta_t \big\rVert_2^2 .
$$

The penalty discourages any client from moving too far from the consensus in a single round, trading some local fit for stability. The strength $\mu$ is a hyperparameter; an un-tuned $\mu$ can under-perform plain FedAvg, a point Chapter 6 returns to.

**FedBN** [9] takes an orthogonal route that is central to this thesis and is developed in Section 2.4: rather than changing the objective, it changes *which parameters are aggregated*, leaving each client's batch-normalization state out of the average so that it can specialize to that client's distribution.

The construction of controlled non-IID partitions for experiments — including the widely used Dirichlet-$\alpha$ scheme [28], [29] — is described in Chapter 5.

## 2.4 Batch normalization and why it is special in federated learning

**Batch normalization (BN)** [6] is a layer inserted between a linear/convolutional operation and its nonlinearity that standardizes activations using statistics computed over the current mini-batch. For a mini-batch $\mathcal{B}$ of pre-activations $\{x_i\}_{i \in \mathcal{B}}$ at a given feature (channel), BN first computes the batch mean and variance,

$$
\mu_{\mathcal{B}} = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} x_i, \qquad
\sigma_{\mathcal{B}}^2 = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} (x_i - \mu_{\mathcal{B}})^2,
$$

then normalizes and applies a learnable affine transformation with scale $\gamma$ and shift $\beta$,

$$
\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon_{\text{BN}}}}, \qquad
y_i = \gamma\, \hat{x}_i + \beta,
$$

where $\epsilon_{\text{BN}}$ is a small constant for numerical stability. Normalizing activations in this way stabilizes and accelerates training. Crucially, BN behaves differently at training and inference time. During training it uses the batch statistics $(\mu_{\mathcal{B}}, \sigma_{\mathcal{B}}^2)$ above. For inference, where a single example has no batch to normalize against, BN uses **running statistics** $(\mu, \sigma^2)$ accumulated during training as exponential moving averages,

$$
\mu \leftarrow (1-m)\,\mu + m\,\mu_{\mathcal{B}}, \qquad
\sigma^2 \leftarrow (1-m)\,\sigma^2 + m\,\sigma_{\mathcal{B}}^2,
$$

with momentum $m$. A BN layer therefore carries two kinds of state: the **learnable affine parameters** $(\gamma, \beta)$, which are updated by gradient descent, and the **running buffers** $(\mu, \sigma^2)$, which are *not* gradient-updated but are accumulated by the forward pass. Both kinds of state encode information about the distribution of activations the layer has seen.

This is exactly what makes BN special in the federated setting. The running statistics of a client's BN layers summarize that client's data distribution. If, under heterogeneity, two clients see very different data, their BN statistics diverge to reflect it. FedAvg blindly averages all parameters and buffers, including BN state, which forces every client to normalize using a single global compromise that fits none of them — a particularly damaging form of client drift.

**FedBN** [9] is the remedy. It partitions the model parameters into a BatchNorm group and a non-BatchNorm group, aggregates the non-BatchNorm parameters by the ordinary FedAvg rule, and **excludes the BatchNorm parameters and running statistics from aggregation entirely**. Each client thus keeps its own BN affine parameters $(\gamma, \beta)$ and its own running buffers $(\mu, \sigma^2)$ across all rounds; the BN layers act as a learned, per-client adapter that calibrates the shared convolutional features to the local data distribution. FedBN was originally validated on feature-shifted image partitions, where it improved over FedAvg. Its mechanism — keeping BatchNorm state local — is both the source of its non-IID advantage and, as Section 2.6 and Chapter 4 show, the source of its incompatibility with off-the-shelf DP-SGD.

## 2.5 Differential privacy and DP-SGD

**Differential privacy (DP)** [5], [23] is a formal, quantitative definition of what it means for an algorithm to protect individual records. The intuition is a stability requirement: the output distribution of the algorithm should be almost unchanged if any one record is added to or removed from the input. Two datasets are called **adjacent** if they differ in exactly one record. A randomized mechanism $\mathcal{M}$ satisfies **$(\varepsilon, \delta)$-differential privacy** if, for every pair of adjacent datasets $D$ and $D'$ and every measurable set $S$ of possible outputs,

$$
\Pr[\mathcal{M}(D) \in S] \;\le\; e^{\varepsilon}\, \Pr[\mathcal{M}(D') \in S] \;+\; \delta .
$$

The **privacy budget** $\varepsilon \ge 0$ bounds how much the presence of a single record can shift the output distribution: smaller $\varepsilon$ means stronger privacy. The parameter $\delta$ is a small probability of failing the $\varepsilon$ bound and is conventionally set far below $1/n$. A central feature of DP is **composition**: the privacy cost of a sequence of DP computations accumulates in a controlled way, which is what allows a guarantee for an iterative training procedure made of many noisy steps.

The basic tool for releasing a real-valued function under DP is the **Gaussian mechanism**. If a function $g$ has $\ell_2$-**sensitivity** $\Delta = \max_{D \sim D'} \lVert g(D) - g(D') \rVert_2$ — the most that one record can change $g$ — then releasing $g(D) + \mathcal{N}(0, \sigma^2 \Delta^2 I)$ satisfies DP with a strength governed by the noise scale $\sigma$. Larger noise buys smaller $\varepsilon$. The art of training under DP is to bound the sensitivity of a gradient step and calibrate the noise accordingly.

**Differentially private SGD (DP-SGD)** [5] does exactly this for stochastic gradient descent. The obstacle is that an ordinary mini-batch gradient has unbounded sensitivity: a single pathological example can make the gradient arbitrarily large, so no fixed amount of Gaussian noise would suffice. DP-SGD bounds the sensitivity by **per-sample gradient clipping**. At each step, for every example $x_i$ in the mini-batch $\mathcal{B}$, it computes the per-sample gradient $g_i = \nabla_\theta \,\ell\big(f_\theta(x_i), y_i\big)$, clips it to a maximum $\ell_2$ norm $C$,

$$
\bar{g}_i = g_i \cdot \min\!\left(1, \frac{C}{\lVert g_i \rVert_2}\right),
$$

so that no single record contributes a gradient of norm greater than $C$, then sums the clipped per-sample gradients and adds Gaussian noise calibrated to that bounded sensitivity:

$$
\hat{g} = \frac{1}{|\mathcal{B}|}\left( \sum_{i \in \mathcal{B}} \bar{g}_i + \mathcal{N}\!\big(0, \sigma^2 C^2 I\big) \right).
\tag{2.1}
$$

The optimizer then takes its step using $\hat{g}$ in place of the ordinary gradient. Two ingredients are doing the work: clipping bounds the contribution of any one record to $C$, and the noise hides that bounded contribution. The **noise multiplier** $\sigma$ controls the trade-off — more noise yields a smaller $\varepsilon$ but a noisier, less useful gradient.

The total privacy cost of running DP-SGD for many steps is tracked by a **privacy accountant**. The tightest practical accountants are based on **Rényi differential privacy (RDP)** [24]. A mechanism is $(\alpha, \rho)$-RDP if, for a Rényi-divergence order $\alpha > 1$, the divergence between its output distributions on adjacent datasets is at most $\rho$; RDP composes by simply *adding* the $\rho$ values across steps, and a final conversion turns the accumulated RDP guarantee into a standard $(\varepsilon, \delta)$-DP guarantee. Because each DP-SGD step operates on a randomly sampled mini-batch, the **privacy amplification by subsampling** further reduces the cost: a record that is not sampled in a given step incurs no privacy loss that step. The accountant takes the sampling rate $q = |\mathcal{B}|/n$, the noise multiplier $\sigma$, and the number of steps $T$, and returns the spent $\varepsilon$. In practice one inverts this: given a target $(\varepsilon, \delta)$, the number of steps, and the sampling rate, the accountant solves for the smallest noise multiplier $\sigma$ that meets the budget. The **Opacus** library [7] implements all of this — per-sample gradient computation, clipping, noise addition, and an RDP accountant — behind a `PrivacyEngine` that wraps an ordinary PyTorch model and optimizer; its `make_private_with_epsilon` entry point performs exactly the target-$(\varepsilon,\delta)$-to-$\sigma$ calibration just described.

## 2.6 The conflict between batch normalization and DP-SGD

The two preceding sections set up a collision that is the technical center of this thesis. DP-SGD's entire guarantee rests on the existence of a well-defined **per-sample gradient** $g_i$ that depends on record $i$ alone — only then does clipping $g_i$ to norm $C$ bound the influence of record $i$, and only then does the calibrated noise yield the claimed $(\varepsilon, \delta)$-DP. Batch normalization violates this premise.

Recall from Section 2.4 that, under BN during training, the normalized activation for sample $i$ is

$$
\tilde{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon_{\text{BN}}}},
$$

where the batch statistics $\mu_{\mathcal{B}}$ and $\sigma_{\mathcal{B}}^2$ are computed over *every* sample in the mini-batch $\mathcal{B}$. The output for sample $i$ therefore depends on all the other samples $j \neq i$ through $\mu_{\mathcal{B}}$ and $\sigma_{\mathcal{B}}^2$. Differentiating the loss, the "per-sample" gradient $g_i$ inherits this dependence: it is a function not only of $(x_i, y_i)$ but of the whole batch. Two consequences follow. First, the object DP-SGD wants to clip is not actually attributable to a single record. Second, and more damagingly for the privacy analysis, removing or replacing one record changes the batch statistics and hence changes *every other sample's* gradient too, so per-sample clipping to norm $C$ no longer bounds the sensitivity of one record's contribution to the aggregate. The clean Gaussian-mechanism argument behind Equation (2.1) collapses.

This is not a subtle numerical issue that can be tuned away; it is structural, and production DP libraries treat it as a hard error. Opacus's `ModuleValidator` inspects a model's architecture when the privacy engine is attached and **refuses to wrap any model containing a `BatchNorm` module**, raising an `UnsupportedModuleError` before training can begin. The library's recommended remedy, exposed as `ModuleValidator.fix(model)`, replaces every BatchNorm layer with **GroupNorm** [8]. GroupNorm normalizes each sample using statistics computed over a group of that sample's own channels, with no dependence on other samples in the batch and no running buffers, so the per-sample gradient is well-defined again and DP-SGD applies cleanly.

The GroupNorm fix is sound for differential privacy and is, in fact, the standard way to train BN-style architectures under DP-SGD. But it is precisely the wrong fix for FedBN. FedBN's non-IID advantage comes entirely from its per-client BatchNorm running statistics — the learnable, distribution-specific buffers that GroupNorm does not have. GroupNorm carries no running statistics, so there is no per-client BN state for FedBN's exclusion rule to keep local, and the personalization mechanism vanishes. The remainder of this thesis is, in essence, about navigating this conflict: keeping BatchNorm — and therefore FedBN's mechanism — alive while still obtaining a record-level $(\varepsilon, \delta)$-DP guarantee, by recognizing that under FedBN the BatchNorm state never crosses the federation boundary and therefore never needs DP protection in the first place. Chapter 4 makes that construction precise.
