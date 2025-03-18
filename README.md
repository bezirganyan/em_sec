# Uncertainty-Aware Set-Valued Classification

## Overview
This repository provides an implementation of a novel approach to set-valued classification that integrates **evidential deep learning** and **subjective logic** to effectively quantify classification uncertainty. The method is designed to be computationally efficient while maintaining high predictive performance, particularly in scenarios where AI-assisted decision-making is crucial.

## Features
- **Set-Valued Classification**: Instead of a single label, the model returns a set of potential labels when uncertainty is high.
- **Evidential Deep Learning**: Uses evidential reasoning to quantify uncertainty in predictions.
- **Dual-Head Architecture**: One head performs multiclass classification, while the other suggests candidate label sets.
- **Computational Efficiency**: Linear worst-case complexity with respect to the number of classes, making it significantly faster than many existing methods.
- **Benchmark Evaluations**: Demonstrates competitive performance while being up to faster than the baseline in inference on benchmark datasets.

## Requirements
This project is built using **PyTorch Lightning** and **Weights & Biases** for logging. Install dependencies using:

```bash
pip install -r requirements.txt
```

## Usage
### Running Training
To train the model on a dataset, use:
```bash
python main.py --dataset cifar10 --model hyper --epochs 300 --enable-wandb
```
Other available datasets:
- `cifar10`
- `cifar100`

### Inference and Testing
To test a trained model:
```bash
python main.py --dataset cifar10 --model cnn --test --ckpt-path path/to/checkpoint.ckpt
```

### WandB Integration
To enable logging in **Weights & Biases**, add `--enable-wandb` and optionally set a custom run name with `--wandb-name my_experiment`.

## Model Selection
This project supports multiple models:
- `cnn`: Standard convolutional model
- `enn`: Evidential neural network [1]
- `beta`: Beta distribution-based multlabel uncertainty model [2]
- `hyper`: EM-SEC model for hyperopinions
- `ds`: Demptser Shafer Evidential Neural Network [3]
- `svp`: Efficient Set-valued prediction model [4]

Example usage:
```bash
python main.py --dataset luma --model enn --epochs 50 --unc-calib
```

## Results and Benchmarks
Our approach achieves **competitive accuracy** while significantly reducing inference time, making it well-suited for real-time AI-assisted decision-making.

For further details, refer to the paper abstract below:

> *In machine learning and deep learning, uncertainty quantification helps to accurately assess a model's confidence in its predictions, enabling the rejection of uncertain outcomes in safety-critical applications. However, in scenarios involving AI-assisted decision-making, proposing multiple plausible decisions can be more beneficial than either not making any decisions or risking incorrect ones. Set-valued classification is a relaxation of standard multiclass classification where, in cases of uncertainty, the classifier returns a set of potential labels instead of a single label. Current methods for set-valued classification often suffer from high computational complexity or fail to adequately quantify uncertainty. In this paper, we introduce a novel, computationally efficient approach to set-valued classification leveraging evidential deep learning and subjective logic, explicitly providing a measure of classification uncertainty. Our method employs a dual-head architecture: one head conducts multiclass evidential classification, while the other suggests candidate label sets when uncertainty is high. The proposed approach has linear worst-case computational complexity with respect to the number of classes. Extensive evaluation on several benchmark datasets demonstrates that our method showcases comparable performance to baseline set-valued methods, while being up to 23 times faster at inference on the benchmark datasets.*

## Citation
If you use this work, please cite:
```
to ba added
```

## References

1. Sensoy, Murat, Lance Kaplan, and Melih Kandemir. "Evidential deep learning to quantify classification uncertainty." Advances in neural information processing systems 31 (2018).
2. Zhao, Chen, et al. "Open set action recognition via multi-label evidential learning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023.
3. Tong, Zheng, Philippe Xu, and Thierry Denoeux. "An evidential classifier based on Dempster-Shafer theory and deep learning." Neurocomputing 450 (2021): 275-293.
4. Mortier, Thomas, et al. "Set-Valued Prediction in Multi-Class Classification." 31st Benelux conference on Artificial Intelligence (BNAIC 2019); 28th Belgian Dutch conference on Machine Learning (Benelearn 2019). Vol. 2491. CEUR, 2019.

