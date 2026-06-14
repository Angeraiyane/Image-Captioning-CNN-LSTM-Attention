# Image Captioning with Visual Attention (CNN-LSTM)
This repository contains the complete implementation and technical benchmark for **Exercise 2** of the Advanced Deep Learning module (Master AIDC). The goal of this multimodal vision-language task is to automatically generate descriptive text from an input image

### Key Features:
* **Data Pipeline:** Custom `COCOCaptionDataset` loader with an optimized tokenization and numericalization vocabulary tracking system.
* **Sequence Sorting (`collate_fn`):** Implementation of a dynamic lot-sorting mechanism based on sequence lengths to efficiently feed PyTorch's `pack_padded_sequence`.
* **Baseline Architecture:** A hybrid encoder-decoder network coupling a deep **ResNet50** convolutional backbone with a multi-layer **LSTM** text generator.
* **Decoding Strategies:** Full comparison between localized **Greedy Decoding** and continuous multi-path **Beam Search Exploration** ($k=3$).
* **Visual Attention Mechanism:** Integration of a spatial **Bahdanau Additive Attention** layer ($7 \times 7$ grid mapping) to dynamically re-align word generation with visual features.
* **NLP Evaluation Metrics:** Automated performance benchmarking tracking Perplexity curves, BLEU-1, BLEU-4, METEOR, and CIDEr metrics.
