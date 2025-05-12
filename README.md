# Grammar Scoring Engine

This project is an advanced grammar scoring engine that analyzes audio files to evaluate speech quality and grammatical accuracy.

## Key Features

- Audio preprocessing and noise reduction
- Speech-to-text conversion (using Whisper model)
- Text analysis and grammar checking
- Machine learning-based scoring system
- Detailed analysis and visualization

## Required Packages

```
pandas
numpy
librosa
soundfile
matplotlib
seaborn
tqdm
torch
transformers
noisereduce
spacy
scikit-learn
tensorflow
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Run the program:
```bash
python grammar_scoring_engine.py
```

2. Input Data:
- `train.csv`: Training data file
- `test.csv`: Test data file

3. Output:
- `submission.csv`: Predicted scores
- `training_history.png`: Training progress visualization
- `predicted_scores_distribution.png`: Score distribution visualization

## Project Structure

- `preprocess_audio()`: Audio file preprocessing
- `extract_audio_features()`: Audio feature extraction
- `transcribe_audio()`: Speech-to-text conversion
- `extract_text_features()`: Text analysis
- `create_model()`: Neural network model creation
- `train_and_evaluate()`: Model training and evaluation

## Features Description

### Audio Processing
- Converts audio to mono
- Resamples to target sample rate
- Removes background noise
- Trims silence

### Feature Extraction
- Duration analysis
- Zero crossing rate
- Energy levels
- MFCC features
- Pitch analysis

### Text Analysis
- Word count and sentence count
- Average sentence length
- Lexical diversity
- POS tag diversity
- Grammar error detection
- Average word length

### Model Architecture
- Deep neural network with multiple layers
- Batch normalization
- Dropout for regularization
- Early stopping
- Learning rate reduction

## License

MIT License 
