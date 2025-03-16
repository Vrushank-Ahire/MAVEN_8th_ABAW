# MAVEN: Multi-modal Attention for Valence-Arousal Emotion Network

This repository contains the implementation of **MAVEN (Multi-modal Attention for Valence-Arousal Emotion Network)**, a novel architecture for dynamic emotion recognition through dimensional modeling of affect. The model integrates visual, audio, and textual modalities via a bi-directional cross-modal attention mechanism, enabling comprehensive interactions between all modality pairs. MAVEN predicts emotions in polar coordinate form (theta and intensity), aligning with psychological models of the emotion circumplex.

![Architecture Diagram](architecture.jpg)

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Results](#results)
7. [Usage](#usage)
8. [Citation](#citation)
9. [License](#license)

## Introduction
MAVEN is designed to recognize emotions in conversational videos by using multi-modal data (visual, audio, and textual). The proposed model employs modality-specific encoders (Swin Transformer for video, HuBERT for audio, and RoBERTa for text) to extract rich feature representations. Our work focuses on the bidirectional cross-modal attention mechanism, which refines each modality's representation through weighted attention from other modalities, followed by self-attention refinement.

## Dataset
The model is trained and evaluated on the **Aff-Wild2** dataset, an audiovisual (A/V) dataset containing 594 videos with approximately 3 million frames from 584 subjects. Each frame is annotated with continuous valence and arousal values, representing emotional states along the dimensions of pleasantness (valence) and intensity (arousal).

## Model Architecture
MAVEN consists of the following components:
1. **Modality-Specific Encoders**:
   - **Visual**: Swin Transformer for capturing local and global visual patterns.
   - **Audio**: HuBERT for extracting acoustic features from raw audio waveforms.
   - **Text**: RoBERTa for linguistic analysis and semantic understanding.

2. **Cross-Modal Attention Mechanism**:
   - Six distinct attention pathways (video-to-audio, video-to-text, audio-to-video, audio-to-text, text-to-video, and text-to-audio) enable bidirectional information flow between modalities.

3. **BEiT Multi-Headed Attention**:
   - After cross-modal fusion, the enhanced features are refined using BEiT-based self-attention to capture global dependencies.

4. **Emotion Prediction**:
   - The final output predicts emotions in polar coordinates (theta and intensity), which are then transformed into valence and arousal values.

## Training
The model is trained using the following setup:
- **Optimizer**: Adam with a learning rate of `1e-4` and weight decay of `1e-4`.
- **Learning Rate Scheduler**: ReduceLROnPlateau with a factor of `0.1` and patience of `5`.
- **Batch Size**: 8.
- **Training Duration**: 100 epochs (patience of 10 epochs).

Pre-trained feature extractors (Swin, HuBERT, RoBERTa, and BEiT-3) are frozen during training to focus optimization on the fusion and prediction layers.

## Evaluation

The performance of the model is evaluated using the *Concordance Correlation Coefficient (CCC)* for both valence and arousal. The overall performance measure $P$ is the average of the CCC values for valence and arousal:  

$$
P = \frac{CCC_{valence} + CCC_{arousal}}{2}
$$


### Baseline Results
The baseline model (pre-trained ResNet-50) achieves the following performance on the validation set: 

$$
\text{CCC}_{\text{valence}} = 0.20
$$

$$
\text{CCC}_{\text{arousal}} = 0.20
$$

$$
P = 0.22
$$

## Results
MAVEN demonstrates superior performance in capturing the complex and nuanced nature of emotional expressions in conversational videos. The model achieves SOTA results on the Aff-Wild2 dataset, significantly outperforming the baseline.

## Usage
To train and evaluate the MAVEN model, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Vrushank-Ahire/MAVEN_8th_ABAW.git
   cd MAVEN
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Aff-Wild2 Dataset**:
   - Ensure you have access to the Aff-Wild2 dataset and place it in the `data/` directory.

4. **Train the Model**:
   ```bash
   python embeddings.py
   python TrainBEiT.py
   python TrainMLP.py 
   ```

5. **Evaluate the Model**:
   ```bash
   python Test.py
   ```

## Citation
If you use this code or the MAVEN model in your research, please cite our paper:

```bibtex
@article{maven2025,
  title={MAVEN: Multi-modal Attention for Valence-Arousal Emotion Network},
  author={Author Name and Second Author},
  journal={Journal Name},
  year={2023},
  publisher={Publisher}
}
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For any questions or issues, please open an issue on GitHub or contact the authors.
