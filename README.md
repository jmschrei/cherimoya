<img src="https://github.com/jmschrei/cherimoya/blob/main/imgs/cherimoya.png">

> [!IMPORTANT]
> Cherimoya is still under active development and may change in ways that are not back compatible. Please make note of the version you are using in case you need to return to it in the future.

Cherimoya is a lightweight genomic sequence-to-function (S2F) model for predicting genomic modalities such as transcription factor binding, chromatin accessibility, and transcription initiation. It builds on concepts that were first introduced by BPNet and ChromBPNet while introducing architectural, algorithmic, and systems-level improvements that improve training stability, efficiency, and predictive performance. Despite needing significantly fewer parameters than other architectures, Cherimoya achieves strong predictive performance across a range of tasks and runs ~5-15x faster when measured on an H200 GPU. 

<img src="https://github.com/jmschrei/cherimoya/blob/main/imgs/cheri-model.png">

The secret to Cherimoya's success is a new Cheri Block, which adapts the ConvNeXT block to the domain of noisy high-throughput genomics experiments. This block is comprised of a dilated depth-wise convolution, a layer norm, a projection into a higher-dimensional space, a GeLU non-linearity, a projection back into the original dimensionality, and then a channel-wise scaling for robustness. Conceptually, this means that the blocks first aggregate information spatially but independently for each feature/channel (the depth-wise convolution) and then aggregate information across features but independently for each position (the two projections). The dilated depth-wise convolution and the layer norm have been fused into an efficient custom GPU kernel that is ~2-3x faster than the native PyTorch implementation.

<img src="https://github.com/jmschrei/cherimoya/blob/main/imgs/cheri-block.png">

### Installation

`pip install cherimoya`

### Key Features

*Lightweight Architecture*: Cherimoya employs a compact convolutional backbone that substantially reduces parameter count while also slightly increasing predictive accuracy. This design enables efficient training, large-scale hyperparameter exploration, interactive usage via browsers, and usage of dozens or hundreds of such models simultaneously in complex design settings.

*Stable training*: Several design choices were made to improve the stability of model training, including the use of layer norm in each layer, a channel-wise scaling on each residual layer that begins at 0.01 (close to the identity mapping), a cosine decay learning rate scheduler with a long warmup (5 epochs by default), removing all bias terms in the Cheri blocks, and, somewhat counterintuitively, removing weight decay from the optimizers.

*Automatic Loss Weight Balancing*: Profile and count losses are combined using learned weighting parameters rather than fixed hyperparameters. This approach replaces the heuristic developed for BPNet and ChromBPNet models and enables the models to scale to larger contexts and across modalities automatically, while also improving gradient stability across datasets with varying signal-to-noise characteristics.

*Muon Optimizer*: Cherimoya uses the Muon optimizer when training the projection layers, and the AdamW optimizer for all other layers and terms. This has significantly accelerated training by reducing the number of epochs needed while modestly improving performance.

*Model Compilation*: Because of the architectural decisions made in the Cheri block, many operations can be automatically fused together in neat ways when using `torch.compile` and so this has been built-in to the forward pass. This seems to offer a ~50-75% speed improvement. Although this compilation needs to only be done once and can be then re-used across models and sessions, it may need to be redone each time the batch size has changed, e.g., for the last batch being processed.

*Mixed Precision*: When data is of reasonable depth, Cherimoya models are best trained using mixed precision, which can offer a ~2x speed improvement (sometimes more when also compiling the model). However, using mixed precision can hurt performance when the data is very low quality or low read depth, such as for TF ChIP-seq experiments or pseudobulks for rare cell types. We recommend using `float32` precision for BPNet-style models as a starting point unless you have particularly high-quality data.


### End-to-End Pipeline

<img src="https://github.com/jmschrei/cherimoya/blob/main/imgs/pipeline.png" width=70%>

Cherimoya provides an integrated command-line pipeline that allows you to go directly from mapped reads, to model training and evaluation, to analysis results. This pipeline improves reproducibility by being self-documenting on the parameter settings for each step, and dramatically reduces the overhead associated with managing seperate tooling for each stage. Specifically, it includes:

- Conversion from BAM/SAM/fragment files to (un)/stranded bigWig(s) using bam2bw
- Peak calling using MACS3
- Calling of GC-matched negatives
- Model training and evaluation
- Attribution scores using *in silico* saturation mutagenesis
- Seqlet calling and annotation using tomtom-lite
- De novo motif discovery using TF-MoDISco

A multi-step pipeline like this has many hyperparameters that can be customized at each step (e.g., number of filters in the model, number of seqlets to use for TF-MoDISco) and requires pointers to several input and output files. Rather than using a giant command-line call, Cherimoya uses JSONs to manage each step of the pipeline. An advantage of using JSONs is that they create a permanent record of the exact command that was run. Although there are many hyperparameters, the user-provided JSONs can be quite small in practice because they are internally merged with the default parameters for each step. The fastest way to begin this process is through the `pipeline-json` command, which takes in pointers to your data files and flags describing the data and produces a valid JSON for the pipeline process. These data files usually include a reference genome, some number of input (and optionally control) BAM/SAM/tsv/tsv.gz files (the `-i` and `-c` arguments can be repeated) a BED file of positive loci, and a MEME formatted motif database used for evaluation of the model.

For example, if you are working with ChIP-seq data that is stranded:

```
cherimoya pipeline-json -s hg38.fa -p peaks.bed.gz -i input1.bam -i input2.bam -c control1.bam -c control2.bam -n test -o pipeline.json -m JASPAR_2024.meme
```

If you are working with ATAC-seq data, which is unstranded and comes in the form of paired-end fragmnents that need to be shifted +4/-4 (as they do in the ChromBPNet work) you can use the following:

```
cherimoya pipeline-json -s hg38.fa -p peaks.bed.gz -i input1.bam -i input2.bam -n atac-test -o atac-pipeline.json -m JASPAR_2024.meme -ps 4 -ns -4 -u -f -pe
```

Note that any of these data pointers can point to remote files. This will stream the data through bam2bw and read the peak files remotely. Processing speed will then depend on the speed of your internet connection and whether the hosting site throttles your connection.

The resulting JSON stored at `pipeline.json` or `atac-pipeline.json` can then be executed using the `pipeline` command. These commands are separated because, although the first command produces a valid JSON that the second command can immediately use, one may wish to modify some of the many parameters in the JSON. These parameters include the number of filters and layers in the model, the training and validation chromosomes, and the p-value threshold for calling seqlets. The defaults for most of these steps seem reasonable in practice, but there is immense flexibility there, e.g., the ability to train the model using a reference genome and then make predictions or attributions on synthetic sequences or the reference genome from another species. In this manner, the JSON serves as documentation for the experiments that have been performed.

```
cherimoya pipeline -p pipeline.json
```

When running the pipeline, a JSON is produced for each one of the steps (except for running TF-MoDISco and annotating the seqlets, which uses `ttl`). Each of these JSONs can be run by itself using the appropriate built-in command. Because some of the values in the JSONs for these steps are set programmatically when running the file pipeline, e.g., the filenames to read in and save to, being able to inspect every one of the JSONs can be handy for debugging.
  

