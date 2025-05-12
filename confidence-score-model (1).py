#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


#importing all basic libraries required in code
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import librosa
import soundfile as sf
from tqdm.notebook import tqdm
import re
import torch
from scipy.stats import pearsonr


# In[3]:


get_ipython().system('pip install librosa pydub openai-whisper transformers sentencepiece torch pandas scikit-learn language-tool-python spacy nltk happytransformer')
get_ipython().system('python -m spacy download en_core_web_sm')


# In[4]:


import transformers
print(transformers.__version__)


# In[5]:


## Create processed audio directory and load training CSV with renamed columns

AUDIO_DIR = '/kaggle/input/shl-intern-hiring-assessment/Dataset/audios/train'
CSV_PATH = '/kaggle/input/shl-intern-hiring-assessment/Dataset/train.csv'
PROCESSED_DIR = '/kaggle/working/processed_audio'
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load CSV and rename columns
train_df = pd.read_csv(CSV_PATH)
train_df.columns = ['filename', 'label']


# In[6]:


## Detect whether an audio file contains speech by analyzing silence, speech ratio, and zero-crossing rate variation

def detect_speech_content(audio_path):
    """Determine if audio contains speech or just instrumental/noise"""
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Multiple detection methods
    non_silence = librosa.effects.split(y, top_db=25)
    speech_ratio = sum(end-start for start, end in non_silence) / len(y) if len(y) > 0 else 0
    
    # Zero-crossing rate variance (speech has more variation than music)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_std = np.std(zcr)
    
    # Combined check
    has_speech = speech_ratio > 0.1 and zcr_std > 0.05
    return has_speech


# In[7]:


# Function: Preprocess audio by resampling, normalizing volume, and trimming silence
def preprocess_audio(file_path, save_path, sr=16000):
    """Enhanced preprocessing with speech detection"""
    y, orig_sr = librosa.load(file_path, sr=None)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr, sr)
    
    # Normalize volume
    y = y / max(abs(y)) if max(abs(y)) > 0 else y
    
    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=25)
    sf.write(save_path, y, sr)
# Function: Detect whether an audio contains speech using silence trimming and ZCR variation
# Process each audio file

# Important: Preprocessing each audio file and detecting speech before transcription
speech_flags = []
for filename in tqdm(train_df['filename']):
    in_path = os.path.join(AUDIO_DIR, filename)
    out_path = os.path.join(PROCESSED_DIR, filename)
    
    # Original preprocessing
    preprocess_audio(in_path, out_path)
    
    # Add speech detection
    has_speech = detect_speech_content(out_path)
    speech_flags.append(has_speech)

# Add speech flag to dataframe
train_df['has_speech'] = speech_flags

# Report number of non-speech files (useful for understanding audio content distribution)
non_speech_count = sum(1 for flag in speech_flags if not flag)
print(f"📊 Detected {non_speech_count}/{len(speech_flags)} files as non-speech/instrumental")

# Check sample rate and duration of a random file
files = os.listdir(PROCESSED_DIR)
print(f"🔎 Found {len(files)} preprocessed audio files.\nExample files:\n", files[:5])

sample_file = os.path.join(PROCESSED_DIR, files[0])
y, sr = librosa.load(sample_file, sr=None)

duration = librosa.get_duration(y=y, sr=sr)
print(f"📁 Sample file: {files[0]}")
print(f"🕒 Duration: {duration:.2f} seconds")
print(f"🎧 Sample rate: {sr} Hz")


# In[8]:


# Load Whisper ASR model for transcription
import whisper

# Load Whisper ASR model
whisper_model= whisper.load_model("base")  # Options: tiny, base, small, medium, large

#  Important: Transcribe only speech-containing files to save compute and avoid noise
transcripts = []

for idx, row in tqdm(train_df.iterrows()):
    fname = row['filename']
    audio_path = os.path.join(PROCESSED_DIR, fname)
    
    if row['has_speech']:
        # Normal transcription for speech files
        result = whisper_model.transcribe(audio_path, language='en')
        transcript = result['text']
    else:
        # Skip transcription for non-speech files
        transcript = ""
    
    transcripts.append(transcript)

# Add transcripts to dataframe
train_df['transcript'] = transcripts

# Save updated CSV
train_df.to_csv('/kaggle/working/train_with_transcripts.csv', index=False)
print("✅ Transcriptions saved to: /kaggle/working/train_with_transcripts.csv")

# Sample transcript review
df = pd.read_csv('/kaggle/working/train_with_transcripts.csv')
print("🧾 Columns:", df.columns.tolist())
print("✅ Total records:", len(df))


# In[9]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")


# In[10]:


FILLERS = ['uh', 'um', 'erm', 'you know', 'like', 'i mean', 'hmm', 'ah', 'uhh', 'huh']

def clean_transcript(text):
    text = text.lower()  # Standard casing
    text = re.sub(r'\b(?:' + '|'.join(FILLERS) + r')\b', '', text)  # Remove fillers
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    text = re.sub(r'\s([?.!,"])', r'\1', text)  # Remove space before punctuation
    text = text.strip()
    return text

# Clean all transcripts
df['cleaned_transcript'] = df['transcript'].astype(str).apply(clean_transcript)

# Save new version
df.to_csv('/kaggle/working/train_cleaned.csv', index=False)
print("✅ Cleaned transcripts saved to: /kaggle/working/train_cleaned.csv")
print(df[['transcript', 'cleaned_transcript']].sample(3))


# In[11]:


get_ipython().system('pip install language-tool-python==2.7.1')


# In[12]:


get_ipython().system('apt-get update')
get_ipython().system('apt-get install -y openjdk-11-jdk')


# In[13]:


import spacy
from happytransformer import HappyTextToText, TTSettings
import re
from tqdm import tqdm
import pandas as pd

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # If model isn't installed, download it
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load text correction model
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
args = TTSettings(
    num_beams=5,
    do_sample=False,
    temperature=1.0,
    top_k=50,
    top_p=1.0
)
def simple_grammar_check(text):
    """Simple rule-based grammar checker to replace LanguageTool"""
    errors = 0
    
    # Check for common grammar issues
    # Double spaces
    errors += len(re.findall(r"  +", text))
    
    # Missing period at end of sentence
    errors += len(re.findall(r"[a-zA-Z]$", text))
    
    # Missing capitalization after period
    errors += len(re.findall(r"\. [a-z]", text))
    
    # Double punctuation
    errors += len(re.findall(r"[,.!?]{2,}", text))
    
    # Common spelling mistakes (very basic)
    common_mistakes = ["teh", "alot", "definately", "occured", "recieve", 
                       "seperate", "wierd", "accomodate", "goverment", "wich"]
    
    for mistake in common_mistakes:
        errors += len(re.findall(r"\b" + mistake + r"\b", text.lower()))
    
    return errors

def extract_enhanced_features(text, has_speech=True):
    """Extract comprehensive linguistic features with special handling for non-speech"""
    features = {}
    
    # Special case for non-speech or empty text
    if not has_speech or not text.strip():
        features['is_empty'] = 1
        features['grammar_errors'] = 0
        features['avg_sentence_length'] = 0
        features['pos_diversity'] = 0
        features['word_count'] = 0
        features['grammar_errors_per_word'] = 0
        features['gec_edits'] = 0
        features['gec_edit_rate'] = 0
        features['lexical_diversity'] = 0
        features['avg_word_length'] = 0
        features['readability_score'] = 0
        return features
    
    # Text is not empty, extract normal features
    doc = nlp(text)
    
    # Regular features
    features['is_empty'] = 0
    features['grammar_errors'] = simple_grammar_check(text)  # Using our custom checker
    
    # Calculate sentence lengths
    sentences = list(doc.sents)
    if sentences:
        sent_lengths = [len(sent) for sent in sentences]
        features['avg_sentence_length'] = sum(sent_lengths) / len(sent_lengths)
    else:
        features['avg_sentence_length'] = 0
    
    pos_tags = [token.pos_ for token in doc if token.pos_ != 'SPACE']
    features['pos_diversity'] = len(set(pos_tags)) if pos_tags else 0

    # Word count and error rate
    words = text.split()
    features['word_count'] = len(words)
    features['grammar_errors_per_word'] = features['grammar_errors'] / max(1, features['word_count'])
    
    # GEC features - grammar correction edits
    if text.strip():
        corrected = happy_tt.generate_text("grammar: " + text).text
        corrected_words = corrected.split()
        
        # Calculate edits
        min_len = min(len(words), len(corrected_words))
        edits = sum(1 for i in range(min_len) if words[i] != corrected_words[i])
        edits += abs(len(words) - len(corrected_words))
        
        features['gec_edits'] = edits
        features['gec_edit_rate'] = edits / max(1, len(words))
    else:
        features['gec_edits'] = 0
        features['gec_edit_rate'] = 0
    
    # Enhanced features
    # Lexical diversity (unique words / total words)
    unique_words = len(set([token.text.lower() for token in doc if token.is_alpha]))
    features['lexical_diversity'] = unique_words / max(1, features['word_count'])
    
    # Average word length
    features['avg_word_length'] = sum(len(w) for w in words) / max(1, len(words))
    
    # Readability score (simplified Flesch)
    features['readability_score'] = 206.835 - (1.015 * features['avg_sentence_length']) - (84.6 * features['avg_word_length'])
    
    return features

# Process all rows in dataframe
all_features = []
for idx, row in tqdm(df.iterrows()):
    text = row['cleaned_transcript']
    has_speech = row['has_speech']
    features = extract_enhanced_features(text, has_speech)
    all_features.append(features)

# Convert to DataFrame and add to main DataFrame
features_df = pd.DataFrame(all_features)
enhanced_df = pd.concat([df, features_df], axis=1)

# Save enhanced features
enhanced_df.to_csv('/kaggle/working/train_features_enhanced.csv', index=False)
print("✅ Enhanced features saved to: /kaggle/working/train_features_enhanced.csv")


# In[14]:


get_ipython().system('pip install happytransformer')


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define features
features = [
    'has_speech',  # Include speech flag
    'grammar_errors', 
    'avg_sentence_length', 
    'pos_diversity',
    'word_count', 
    'grammar_errors_per_word',
    'gec_edits', 
    'gec_edit_rate',
    'lexical_diversity',
    'avg_word_length',
    'readability_score'
]

X = enhanced_df[features]
y = enhanced_df['label']

# Split data for training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature-based models
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_lgb = lgb.LGBMRegressor(n_estimators=100, random_state=42)
model_ridge = Ridge(alpha=1.0)

# Train models
model_rf.fit(X_train, y_train)
model_lgb.fit(X_train, y_train)
model_ridge.fit(X_train, y_train)

# Predict on validation
pred_rf = model_rf.predict(X_val)
pred_lgb = model_lgb.predict(X_val)
pred_ridge = model_ridge.predict(X_val)

# Ensemble (averaged predictions)
ensemble_feat_preds = (pred_rf + pred_lgb + pred_ridge) / 3

# Evaluation
mae = mean_absolute_error(y_val, ensemble_feat_preds)
rmse = np.sqrt(mean_squared_error(y_val, ensemble_feat_preds))
corr, _ = pearsonr(y_val, ensemble_feat_preds)

print(f"📊 Feature Ensemble MAE: {mae:.3f}")
print(f"📉 Feature Ensemble RMSE: {rmse:.3f}")
print(f"🔗 Feature Ensemble Pearson Correlation: {corr:.3f}")


# In[16]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os

os.environ["WANDB_DISABLED"] = "true"

# Prepare text data for transformer model
text_df = enhanced_df[['cleaned_transcript', 'label', 'has_speech']]
text_df = text_df.rename(columns={'cleaned_transcript': 'text'})

# Split for transformer model
train_text, val_text = train_test_split(text_df, test_size=0.2, random_state=42)

# Create HuggingFace datasets
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_hf = Dataset.from_pandas(train_text[['text', 'label']])
val_hf = Dataset.from_pandas(val_text[['text', 'label']])
train_hf = train_hf.map(tokenize)
val_hf = val_hf.map(tokenize)
train_hf = train_hf.rename_column("label", "labels")
val_hf = val_hf.rename_column("label", "labels")

# DistilBERT for regression
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=1
)

# Metrics calculation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.squeeze()
    mse = ((preds - labels) ** 2).mean()
    mae = np.abs(preds - labels).mean()
    corr = np.corrcoef(preds, labels)[0, 1]
    return {"mae": mae, "mse": mse, "pearson": corr}

# Training arguments for version 4.51.1
# Training arguments for version 4.51.1
args = TrainingArguments(
    output_dir="./bert-regressor",
    eval_steps=25,
    logging_steps=25,
    save_steps=0,  # equivalent to save_strategy="no"
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=12,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    disable_tqdm=False,
    report_to="none",  # Using string "none" instead of None
    dataloader_pin_memory=False,
)

# Set device
if torch.cuda.is_available():
    model.to("cuda")
    print("✅ Model on GPU")

# Create trainer and train
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_hf,
    eval_dataset=val_hf,
    compute_metrics=compute_metrics,
)

trainer.train()

# Get transformer predictions on validation set
bert_preds_val = trainer.predict(val_hf).predictions.squeeze()


# In[17]:


# ======== META ENSEMBLE ========

from sklearn.linear_model import LinearRegression

# Stack predictions from both model types
stacked_val = np.vstack([bert_preds_val, ensemble_feat_preds]).T

# Train meta-regressor
meta_model = LinearRegression()
meta_model.fit(stacked_val, y_val)

# Final predictions with meta-ensemble
final_val_preds = meta_model.predict(stacked_val)

# Handle non-speech cases specially
for i, has_speech in enumerate(val_text['has_speech'].values):
    if not has_speech:
        # Calculate default score for non-speech from training
        nonspeech_default = enhanced_df[~enhanced_df['has_speech']]['label'].median() if sum(~enhanced_df['has_speech']) > 0 else 3
        final_val_preds[i] = nonspeech_default

# Final evaluation
mae = mean_absolute_error(y_val, final_val_preds)
rmse = np.sqrt(mean_squared_error(y_val, final_val_preds))
pearson = pearsonr(y_val, final_val_preds)[0]

print(f"📊 Final Meta-Ensemble MAE: {mae:.3f}")
print(f"📉 Final Meta-Ensemble RMSE: {rmse:.3f}")
print(f"🔗 Final Meta-Ensemble Pearson: {pearson:.3f}")
# === Save all trained models ===

import joblib
import torch

# Save feature-based models
joblib.dump(model_rf, 'random_forest_model.pkl')
joblib.dump(model_lgb, 'lgbm_model.pkl')
joblib.dump(model_ridge, 'ridge_model.pkl')

# Save Transformer (BERT) model
torch.save(model.state_dict(), 'bert_regressor.pth')

# Save Meta model
joblib.dump(meta_model, 'meta_ensemble_model.pkl')

print("✅ All models saved successfully!")


# In[18]:


# Define test paths
TEST_AUDIO_DIR = '/kaggle/input/shl-intern-hiring-assessment/Dataset/audios/test'
TEST_CSV_PATH = '/kaggle/input/shl-intern-hiring-assessment/Dataset/test.csv'
TEST_PROCESSED_DIR = '/kaggle/working/processed_test_audio'
os.makedirs(TEST_PROCESSED_DIR, exist_ok=True)

# Load test data
test_df = pd.read_csv(TEST_CSV_PATH)

# Process test audio
speech_flags_test = []
for filename in tqdm(test_df['filename']):
    in_path = os.path.join(TEST_AUDIO_DIR, filename)
    out_path = os.path.join(TEST_PROCESSED_DIR, filename)
    
    preprocess_audio(in_path, out_path)
    has_speech = detect_speech_content(out_path)
    speech_flags_test.append(has_speech)

test_df['has_speech'] = speech_flags_test

# Transcribe test audio
transcripts_test = []
for idx, row in tqdm(test_df.iterrows()):
    fname = row['filename']
    audio_path = os.path.join(TEST_PROCESSED_DIR, fname)
    
    if row['has_speech']:
        result = whisper_model.transcribe(audio_path, language='en')
        transcript = result['text']
    else:
        transcript = ""
        
    transcripts_test.append(transcript)

test_df['transcript'] = transcripts_test

# Clean test transcripts
test_df['cleaned_transcript'] = test_df['transcript'].apply(clean_transcript)

# Save cleaned test data
test_df.to_csv('/kaggle/working/test_cleaned.csv', index=False)
print("✅ Cleaned test transcripts saved.")



# In[19]:


# Extract features for test set
test_features = []
for idx, row in tqdm(test_df.iterrows()):
    text = row['cleaned_transcript']
    has_speech = row['has_speech']
    features = extract_enhanced_features(text, has_speech)
    test_features.append(features)
# Convert to DataFrame and merge with test data
test_features_df = pd.DataFrame(test_features)
test_enhanced_df = pd.concat([test_df, test_features_df], axis=1)


# In[20]:


# ======== PREDICT ON TEST SET ========

# Prepare test data for transformer model
test_hf = Dataset.from_pandas(test_df[['cleaned_transcript']].rename(columns={"cleaned_transcript": "text"}))
test_hf = test_hf.map(tokenize)

# Get transformer predictions
bert_test_preds = trainer.predict(test_hf).predictions.squeeze()

# Get feature-based predictions
# Match training features exactly
training_feature_cols = model_rf.feature_names_in_  # Automatically stores features seen at .fit()
X_test_feat = test_enhanced_df[training_feature_cols]


pred_rf_test = model_rf.predict(X_test_feat)
pred_lgb_test = model_lgb.predict(X_test_feat)
pred_ridge_test = model_ridge.predict(X_test_feat)
ensemble_feat_test_preds = (pred_rf_test + pred_lgb_test + pred_ridge_test) / 3

# Stack and apply meta-regressor
stacked_test_preds = np.vstack([bert_test_preds, ensemble_feat_test_preds]).T
final_test_preds = meta_model.predict(stacked_test_preds)

# Special handling for non-speech files
nonspeech_default = enhanced_df[~enhanced_df['has_speech']]['label'].median() if sum(~enhanced_df['has_speech']) > 0 else 3
for i, has_speech in enumerate(test_df['has_speech']):
    if not has_speech:
        final_test_preds[i] = nonspeech_default

test_df['label'] = (np.round(final_test_preds * 2) / 2).clip(0, 5)


# In[21]:


# ======== GENERATE SUBMISSION ========

submission = test_df[['filename', 'label']]
submission.to_csv('/kaggle/working/submission.csv', index=False)


# In[ ]:




