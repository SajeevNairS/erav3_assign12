# GPT Model Training Project

This project implements a PyTorch-based GPT (Generative Pre-trained Transformer) model training pipeline. The implementation is based on the GPT-2 architecture, specifically the 124M parameter version, with optimizations for efficient training and convergence.

## Features

- Implementation of GPT-2 architecture (124M parameters)
- Efficient training pipeline with gradient clipping and learning rate scheduling
- Early stopping when target loss is achieved
- Automatic device selection (GPU/CPU)
- Model checkpointing to save best performing models
- Progress tracking with detailed logging

## Project Structure

```
.
├── train_get2_8_init.py    # Core GPT model implementation
├── train.py                # Training script
├── requirements.txt        # Project dependencies
├── .gitignore             # Git ignore rules
└── input.txt              # Training data (not tracked in git)
```

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster training)
- At least 16GB RAM (32GB recommended)
- Training data in text format

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
# On Unix/macOS
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your training data:
   - Place your training text data in a file named `input.txt`
   - The text should be preprocessed and ready for training

2. Run the training script:
```bash
python train.py
```

The training script will:
- Automatically detect and use GPU if available
- Train the model until either:
  - The target loss (< 0.099999) is achieved
  - The maximum number of steps (10,000) is reached
- Save the best model checkpoint to `best_model.pt`
- Display training progress and metrics every 10 steps

## Training Parameters

- Model Size: 124M parameters
- Batch Size: 32
- Sequence Length: 64
- Learning Rate: 1e-4
- Weight Decay: 0.1
- Gradient Clipping: 1.0
- Maximum Steps: 10,000
- Early Stopping: When loss < 0.099999

## Output

The training process will generate:
- Console output showing training progress
- Model checkpoints saved as `best_model.pt`
- Training logs showing loss and learning rate

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Implementation based on the GPT-2 architecture
- Uses tiktoken for tokenization
- Built with PyTorch
