# Sarcasm Detection Using Transformers

## Overview
This project implements a text classification model using transformer-based architectures (BERT/DistilBERT) to detect sarcasm in news headlines. The model is trained on the Sarcasm Headlines Dataset to distinguish between sarcastic and non-sarcastic headlines.

## Features
- 🔍 Advanced text preprocessing with NLTK
- 🤖 Transformer-based model implementation
- 📊 Comprehensive evaluation metrics
- 📈 Performance visualization tools
- 🔄 Easy-to-use training pipeline

## Requirements

### Python Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- transformers>=4.30.0
- torch>=2.0.0
- scikit-learn>=1.0.0
- pandas>=1.5.0
- numpy>=1.23.0
- matplotlib>=3.5.0
- seaborn>=0.12.0
- nltk>=3.8.0
- jupyter>=1.0.0

## Dataset
The Sarcasm Headlines Dataset contains:
- News headlines from various sources
- Binary labels (1: sarcastic, 0: non-sarcastic)

### Structure
```
headline            | is_sarcastic
--------------------|-------------
Sample headline 1   | 1
Sample headline 2   | 0
```

## Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/sarcasm-detection.git
cd sarcasm-detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage
```python
from sarcasm_classifier import SarcasmClassifier

# Initialize the classifier
classifier = SarcasmClassifier(model_name='distilbert-base-uncased')

# Preprocess and predict
headline = "Local man wins lottery, decides to keep working anyway"
prediction = classifier.predict(headline)
```

### 3. Training a New Model
```python
# Load your data
train_data = pd.read_csv('path/to/train.csv')

# Train the model
classifier.train(
    train_texts=train_data['headline'],
    train_labels=train_data['is_sarcastic'],
    epochs=3,
    batch_size=16
)
```

## Model Performance
Current model metrics on test set:
- Accuracy: 76%
- F1-Score: 78%
- Precision: 75%
- Recall: 77%

## Project Structure
```
sarcasm-detection/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── saved_models/
├── notebooks/
│   └── training.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Advanced Usage

### Custom Preprocessing
```python
classifier = SarcasmClassifier()

# Add custom preprocessing steps
def custom_preprocess(text):
    # Your custom preprocessing logic
    return processed_text

classifier.preprocess_fn = custom_preprocess
```

### Model Fine-tuning
```python
classifier.train(
    train_texts=texts,
    train_labels=labels,
    learning_rate=2e-5,
    epochs=5,
    batch_size=16,
    warmup_steps=500
)
```

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- HuggingFace Transformers library
- Sarcasm Headlines Dataset creators
- NLTK team for text processing tools

## Contact
Your Name - your.email@example.com
Project Link: https://github.com/yourusername/sarcasm-detection

## Citation
If you use this project in your research, please cite:
```
@software{sarcasm_detection,
  author = {Your Name},
  title = {Sarcasm Detection Using Transformers},
  year = {2024},
  url = {https://github.com/yourusername/sarcasm-detection}
}
```
