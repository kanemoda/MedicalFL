<!-- FRONT MATTER — the only front-matter item produced in this draft.
     Header block follows the Ege University English thesis template
     (Title / Surname, Name / Degree / Supervisor / Date / Pages / body / Keywords).
     Date and page count are admin fields left as placeholders for the author. -->

# Abstract

**WHEN DOES FEDERATED BATCHNORM HELP UNDER DIFFERENTIAL PRIVACY? AN EVAL-REGIME CHARACTERIZATION OF DP-FEDBN ON MEDICAL ECG CLASSIFICATION**

Bağlar, Efe Deniz

B.Sc. in Computer Engineering

Supervisor: Prof. Dr. Hasan Bulut

[TODO: Month Year], [TODO: XX] Pages

Federated learning (FL) for medical artificial intelligence must reconcile two competing demands: per-site personalized performance and rigorous differential-privacy (DP) guarantees. Federated BatchNorm (FedBN) achieves the former by keeping BatchNorm statistics local to each client, but it is incompatible with differentially private stochastic gradient descent (DP-SGD) through standard libraries such as Opacus, because BatchNorm couples samples within a batch and breaks the per-sample gradient accounting that DP-SGD requires. As a result, most differentially private FL deployments fall back to BatchNorm-replacement workarounds — typically GroupNorm — that remove the very mechanism on which FedBN's non-IID advantage depends.

This thesis formalizes and empirically characterizes **DP-FedBN**, a dual-optimizer composition that applies DP-SGD to the non-BatchNorm parameters while keeping the BatchNorm parameters and running statistics local and unprotected, since they never cross the federation boundary. The term "DP-FedBN" already appears as a single-paragraph, party-level DP baseline (Laplace noise on aggregated model updates) in ADCOL (ICML 2023); to our knowledge, this work is the first **record-level DP-SGD** composition of FedBN's BatchNorm-locality with explicit treatment of the Opacus engineering obstacle, together with the first systematic medical empirical characterization of that composition across heterogeneity regimes and privacy budgets.

We benchmark FedBN, DP-FedAvg+GroupNorm, and DP-FedBN on MIT-BIH five-class beat classification and PTB-XL binary record classification across six non-IID partition regimes and three privacy budgets (about forty configurations). Two findings emerge. First, the *eval-regime flip* is task-agnostic: FedBN ranks worst under pooled-central evaluation (losing by up to 0.31 macro F1) while ranking best under per-client local evaluation (winning by up to +0.72 macro F1), confirmed on both datasets. Second, DP-FedBN's local advantage is partition-dependent: it preserves a +0.17 local-F1 edge over GroupNorm on label-skew at ε = 3, but loses under Dirichlet skew on both datasets. We provide a heterogeneity-typed decision rubric for medical FL practitioners.

**Keywords:** federated learning, differential privacy, batch normalization, ECG classification, non-IID data, Opacus, medical AI.
