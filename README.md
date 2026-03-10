<img src="https://github.com/jmschrei/cherimoya/blob/main/imgs/cherimoya.png">

> [!IMPORTANT]
> Cherimoya is still under active development and may change in ways that are not back compatible. Please make note of the version you are using in case you need to return to it in the future.

Cherimoya is a lightweight genomic sequence-to-function (S2F) model for predicting genomic modalities such as transcription factor binding, chromatin accessibility, and transcription initiation. It builds on concepts that were first introduced by BPNet and ChromBPNet while introducing architectural, algorithmic, and systems-level improvements that improve training stability, efficiency, and predictive performance. Despite needing significantly fewer parameters than other architectures, Cherimoya achieves strong predictive performance across a range of tasks and runs ~5-15x faster when measured on an H200 GPU. 

<img src="https://github.com/jmschrei/cherimoya/blob/main/imgs/cheri-model.png">

The secret to Cherimoya's success is a new Cheri Block, which adapts the ConvNeXT block to the domain of noisy high-throughput genomics experiments. This block is comprised of a dilated depth-wise convolution, a layer norm, a projection into a higher-dimensional space, a GeLU non-linearity, a projection back into the original dimensionality, and then a channel-wise scaling for robustness. Conceptually, this means that the blocks first aggregate information spatially but independently for each feature/channel (the depth-wise convolution) and then aggregate information across features but independently for each position (the two projections). The dilated depth-wise convolution and the layer norm have been fused into an efficient custom GPU kernel that is ~2-3x faster than the native PyTorch implementation.

<img src="https://github.com/jmschrei/cherimoya/blob/main/imgs/cheri-block.png">

--

### Installation

`pip install cherimoya`

---

## Key Features

### Lightweight Architecture
Cherimoya employs a compact convolutional backbone that substantially reduces parameter count without sacrificing predictive accuracy. This design enables efficient training, large-scale hyperparameter exploration, interactive usage via browsers, and usage of dozens or hundreds of such models simultaneously in complex design settings.

### Automatic Loss Weight Balancing
Profile and count losses are combined using learned weighting parameters rather than fixed hyperparameters. This approach replaces the heuristic developed for BPNet and ChromBPNet models and enables the models to scale to larger contexts and across modalities automatically, while also improving gradient stability across datasets with varying signal-to-noise characteristics.

### Muon Optimizer
Cherimoya uses the Muon optimizer when training the projection layers, and the AdamW optimizer for all other layers. This has significantly accelerated training by reducing the number of epochs needed while modestly improving performance.

### End-to-End Pipeline
Cherimoya provides an integrated pipeline covering:

- BAM/SAM/fragment file conversion using bam2bw
- Peak calling using MACS3
- model training
- evaluation
- downstream analysis and motif discovery using TF-MoDISco

<img src="https://github.com/jmschrei/cherimoya/blob/main/imgs/pipeline.png">
  

This design supports reproducible end-to-end experiments and reduces the overhead associated with managing separate tooling for each stage.

