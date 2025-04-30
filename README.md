# BlinkWise [MobiSys'25]

The official repo of BlinkWise: Tracking Blink Dynamics and Mental States on Glasses.

## Updates

- **2025-04-30**: Initial release of the BlinkWise codebase.

## Installation

Please first clone the repository and navigate to the project root:

```bash
git clone https://github.com/penn-waves-lab/BlinkWise.git
cd BlinkWise
```

All following instructions assume you are in the project root directory.

We offer installation options using **Conda/pip** or **Docker**.

### Using Conda

First, create a Conda environment and install the required dependencies:

```bash
conda create -n blinkwise python=3.9
conda activate blinkwise

pip install -r requirements.txt
```

BlinkWise requires TensorFlow 2.15.0.post1.
Please install the GPU version of TensorFlow along with CUDA dependencies:

```bash
pip install 'tensorflow[and-cuda]==2.15.0.post1'
conda install -y -c nvidia/label/cuda-12.1.0 cuda
```

_Optional_: Install the following package to enable model architecture visualization:

```bash
conda install -y -c conda-forge graphviz 
```

### Using Docker

We require NVIDIA GPUs to use docker.

1. Install Docker for your platform by following
   the [official installation guide](https://docs.docker.com/engine/install/).

2. To enable GPU support, install the NVIDIA Container Toolkit. Follow the instructions
   provided [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

3. Build the Docker image (9.2 GB) and start the container using the following commands:

```bash
docker build -t blinkwise .
docker run -it --gpus all -v $(pwd):/app -p 8888:8888 blinkwise
```

If you have encountered permission issues, temporary fix is to add `sudo` before the `docker` commands. Or
see [Linux post-installation steps for Docker Engine
](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user) for a permanent fix.

By default, the container will start in the `/app` directory. Please stay in this directory to run the following
commands.

If you plan to use the container to run Jupyter notebooks in the following steps, you can start the Jupyter server by
running the following command inside the container:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Then, open the provided link in your browser to access the Jupyter notebook interface. It will be a link like:

```
http://127.0.0.1:8888/lab?token=<TOKEN>
```

## Usage

### Dataset Preparation

#### Raw Dataset

> **NOTE**: If the goal is to test out training and evaluation scripts, you can skip the raw dataset and download the
> processed dataset directly below.

Due to IRB restrictions, we cannot provide video recordings from the experiments. However, we include processed key
facial landmarks and corresponding eye openness measurements as the ground truth.

To download the raw experimental dataset (~132.5 GB), run:

```bash
python scripts/download.py --raw-dataset
```

The default download path is `data/raw-dataset`. Manual download is
available at [PennBox](https://upenn.box.com/v/blinkwise-raw-dataset) and
please place the included folder `raw-dataset/` under `data/`.

#### Processed Dataset

You may download the processed dataset (~17.2 GB) by running:

```bash
python scripts/download.py --processed-dataset
```

The default download path is `data/processed-dataset`. It is also available at
PennBox [here](https://upenn.box.com/v/blinkwise-processed-dataset) for manual download and please place the included folder `processed-dataset/` under `data/`.

Dataset structure can be found in [`data/README.md`](data/README.md).

Alternatively, to try out the processing script, run:

```bash
python scripts/construct_data.py \
    --data-folder data/raw-dataset \
    --output-folder data/processed-dataset \
    -v
```

The raw measurements will be processed using the following protocol:

1. Radar signals are queried at specified range bins (2.5 to 5.5 cm), then filtered with a differentiator and a low-pass
   filter.
2. Eye openness from video is smoothed and normalized to a range of 0 to 1.

### Training

Pre-trained artifacts are available for download (~76 MB) by running:

```bash
python scripts/download.py --pretrained-artifacts
```

The default download path is `data/reproducing-results`. Or you may manually download
at [here](https://upenn.box.com/v/blinkwise-checkpoints) (preview available).
Please place the included folder `reproducing-results/` under `data/`.

To train the model, run:

```bash
python scripts/train.py -c configs/example_config.json
```

Before training, if you placed `processed-dataset` in a different location or want to save the results elsewhere, update
`configs/example_config.json` with your settings by modifying the `data_folder` and `output_root_folder` fields. See [
`src/models/config/experiment_config.py`](src/models/config/experiment_config.py) for a detailed explanation of field
names in the config.

By default, trained artifacts will be saved as `data/reproducing-results/unet_YYYYMMDD_HHMMSS`.

Artifacts structure can be found in [`data/README.md`](data/README.md).

#### Model Optimization

We use quantization-aware training (QAT) to reduce the model’s performance drop after quantization.

If you have downloaded the pre-trained artifacts from the previous step, it already includes the QATd and quantized
model.

To try out QAT by yourself, run:

```bash
python scripts/qat.py -r data/reproducing-results/unet_REPLACE_WITH_YOUR_OWN
```

This script fine-tunes a pre-trained model with QAT. You can adjust the learning rate and the number of epochs using the
`--finetune-lr` and `--finetune-epochs` flags.

### Evaluation

To evaluate the float-point-precision model, run:

```bash
python scripts/evaluate.py \
    -r data/reproducing-results/unet_20241203_214444 \
    -d data/processed-dataset
```

Replace `unet_20241203_214444` if you prefer to use your own trained model.

To evaluate the quantized model, add the `--tflite` flag:

```bash
python scripts/evaluate.py \
    -r data/reproducing-results/unet_20241203_214444 \
    -d data/processed-dataset \
    --tflite
```

Similarly, replace `unet_20241203_214444` if you prefer to use your own quantized model.

The following results in the paper can be reproduced:
1. Table 2: Blink Detection Performance
2. Table 3: Blink Phase Analysis Performance
3. metrics (correlation and errors) in the paragraph _Openness Prediction_ under Sec 6.2 

### Recurrentization baselines

Recurrentization is an additional step to improve the model’s memory efficiency and latency. This process decomposes the
model into smaller chunks and caches intermediate results to reduce computational overhead.

In our implementation, the first two levels of the U-Net-like model are divided into smaller recurrentized models, each
encapsulating convolutional blocks. Hidden state management is not included but can be implemented easily.

To explore recurrentization, refer to the notebook [
`notebooks/recurrentization.ipynb`](notebooks/reccurentization.ipynb), which reproduces the following results from the
paper:

1. Figure 5: Network layer-wise memory profile.
2. Figure 11 (b): Analytical computation overhead (FLOPs) comparison among the original model,
   the [patch-to-patch inference](https://arxiv.org/abs/2110.15352v2) model, and the recurrentized model.

### Case studies

To perform the same set of statistical tests mentioned in the paper, please first download the following datasets by
running:

```bash
python scripts/download.py --case-studies
```

The default download path is `data/case-studies`.
Or download manually from: [here](https://upenn.box.com/v/blinkwise-case-studies) (preview available).
Please place the included folder `case-studies/` under `data/`.

Fields of csv files can be found in [`data/README.md`](data/README.md). 

The notebook [`notebooks/case_studies.ipynb`](notebooks/case_studies.ipynb) reproduces the following results from the paper:

1. Table 4: Correlations between Drowsiness Measures and Blink Parameters
2. Figure 16: Drowsiness evaluation over 12 hours.
3. Table 5: Blink Parameter Variations Across Visual Task Difficulty Levels
4. Figure 18: Blink parameters across different workloads. Abbreviations are the same as in Table 5.
5. Figure 19: Partial blink detection.

## Code Structure

For customization, the following is an overview of the code structure:

```
src
├── core                      (core utilities: logging, constants, file paths)
├── data                      (dataset processing)
│ ├── dataset
│ ├── event_proposal      (Section 4.3: event proposal algorithm)
│ └── signal_processing   (Section 3.2: efficient background mitigation)
├── evaluation
│ ├── labeling            (Section 5: blink phase segmentation)
│ ├── metrics             (Section 6.2: blink dynamics performance)
│ ├── reconstruction      (openness prediction)
│ └── utils
├── models
│ ├── config              (dataset, model, and training configurations)
│ ├── data                (data loaders)
│ ├── networks            (model architectures)
│ ├── training            (training utilities)
│ └── utils
└── optimization              (QAT and recurrentization)
```

For further details, refer to the code documentation.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Citation

If you find this work useful, please consider citing:

```
@inproceedings{BlinkWise,
  author = {Hu, Dongyin and Yang, Xin and Yuh, Ahhyun and Wang, Zihao and Al-Aswad, Lama A. and Lee, Insup and Zhao, Mingmin},
  title = {Tracking Blink Dynamics and Mental States on Glasses},
  booktitle = {Annual International Conference on Mobile Systems, Applications and Services (MobiSys)},
  year = {2025},
  doi = {10.1145/3711875.3729150},
}
```