# Filipino Text Classification Project - AI Agent Instructions

## Project Overview
This is a **Filipino language NLP research project** focused on hate speech and text classification using transformer models. The project combines multiple Filipino text datasets with fine-tuning capabilities for BERT/ELECTRA models from the `Filipino-Text-Benchmarks` repository.

## Architecture & Key Components

### Core Structure
```
├── datasetImportLib.py          # HuggingFace dataset loading (mteb/FilipinoHateSpeechClassification)
├── Filipino-Text-Benchmarks-master/  # Submodule: training infrastructure
│   ├── train.py                 # Main training script with extensive CLI args
│   └── utils/
│       ├── data.py              # Dataset preprocessing & tokenization
│       └── training.py          # Training loops, evaluation, and model management
├── hatespeech/                  # Local hate speech dataset (train/valid/test splits)
├── filipino-tiktok-hatespeech-main/  # TikTok hate speech dataset
└── *.csv                        # Additional datasets (cyberbullying, aggression)
```

### Data Flow
1. **Dataset sources**: HuggingFace Hub (`hf://datasets/...`), local CSV files with `text,label` format
2. **Preprocessing**: `utils/data.py` handles tokenization, creates TensorDatasets with caching (MD5-hashed filenames)
3. **Models**: Pre-trained Filipino BERT/ELECTRA from HuggingFace (`jcblaise/electra-tagalog-*`, `jcblaise/bert-tagalog-*`)

## Critical Developer Workflows

### Running Fine-tuning (Primary Workflow)
The project uses `Filipino-Text-Benchmarks-master/train.py` as the main entry point:

```bash
# Standard hate speech classification
python Filipino-Text-Benchmarks-master/train.py \
    --pretrained jcblaise/electra-tagalog-small-cased-discriminator \
    --train_data hatespeech/train.csv \
    --valid_data hatespeech/valid.csv \
    --test_data hatespeech/test.csv \
    --checkpoint finetuned_model \
    --msl 128 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --epochs 3 \
    --add_token [LINK],[MENTION],[HASHTAG]
```

**Key parameters to adjust:**
- `--data_pct`: Simulate low-resource settings (default: 1.0)
- `--text_columns`: Comma-separated for entailment tasks (e.g., `s1,s2`)
- `--label_columns`: Comma-separated for multi-label classification
- `--add_token`: Special tokens for social media text (brackets required)

### HuggingFace Authentication
**CRITICAL**: Before loading datasets from HuggingFace Hub, authenticate:
```powershell
huggingface-cli login  # Or: hf auth login
```
This is required for `datasetImportLib.py` which uses `hf://datasets/` protocol.

### Hyperparameter Tuning
Uses Weights & Biases sweeps (see `sample_sweep.yaml`):
```bash
wandb sweep -p PROJECT_NAME Filipino-Text-Benchmarks-master/sample_sweep.yaml
wandb agent USERNAME/PROJECT_NAME/SWEEP_ID
```

## Project-Specific Conventions

### Dataset Format Requirements
- **Binary classification**: CSV with `text,label` columns (label: 0/1)
- **Multi-label**: Multiple label columns (e.g., `absent,dengue,health,mosquito,sick`)
- **Sentence pairs**: Two text columns (e.g., `s1,s2`) for entailment tasks
- **Social media preprocessing**: Use `--add_token [LINK],[MENTION],[HASHTAG]` for Twitter/TikTok data

### Model Selection Pattern
- **Cased vs Uncased**: Use cased for proper nouns, uncased for general text
- **Size**: Small (14M params) for quick experiments, Base (110M params) for final results
- **ELECTRA discriminator**: Primary choice for classification (more efficient than BERT generators)

### Caching Strategy
- Training cache: `cache_<MD5>.pt` files auto-generated from data paths + hyperparameters
- Control with `--save_cache true/false` and `--retokenize_data true/false`
- Cache keys include: data paths, MSL, seed, pretrained model name, data_pct

### FP16 Training
Requires NVIDIA Apex library:
```bash
--fp16 true --opt_level O1  # Mixed precision training
```
Training code checks `APEX_AVAILABLE` and falls back to FP32 gracefully.

## Dependencies & Environment

**Core libraries** (inferred from code):
- PyTorch 1.x with CUDA support
- HuggingFace Transformers 4.0.0+
- pandas, numpy, tqdm
- Optional: Weights & Biases, pytorch-lamb, apex

**GPU Requirements**: Experiments designed for P100/V100 GPUs

## Integration Points

### HuggingFace Models Hub
All pretrained models use `jcblaise/*` namespace:
- Load with: `AutoModelForSequenceClassification.from_pretrained('jcblaise/electra-tagalog-...')`
- Save checkpoints: Compatible with HuggingFace Transformers pipeline

### Model Architecture Detection
Code checks model type via string matching:
```python
if 'distilbert' in str(type(model)) or 'roberta' in str(type(model)):
    # No token_type_ids for these architectures
```

### Optimizer Support
- **AdamW** (default): Weight decay applied selectively (no decay on bias/LayerNorm)
- **LAMB**: Requires `pytorch-lamb` package, triggered by `--optimizer lamb`

## Common Pitfalls

1. **CSV line terminators**: Always use `lineterminator='\n'` with `pd.read_csv()` for Filipino text
2. **Token limits**: MSL (max sequence length) defaults to 128; increase for longer texts
3. **Special tokens**: Must match training data preprocessing (e.g., `[LINK]` not `<LINK>`)
4. **Model vs Checkpoint**: `--pretrained` loads base models, `--checkpoint` is output path for fine-tuned weights
5. **Boolean arguments**: Use `true/false` strings, not Python booleans (custom `str2bool` parser)

## Example Workflows

**Quick test with 10% data:**
```bash
python Filipino-Text-Benchmarks-master/train.py \
    --pretrained jcblaise/electra-tagalog-small-uncased-discriminator \
    --train_data hatespeech/train.csv --valid_data hatespeech/valid.csv --test_data hatespeech/test.csv \
    --data_pct 0.1 --epochs 1 --checkpoint quick_test
```

**Multi-label classification:**
```bash
--label_columns absent,dengue,health,mosquito,sick  # No spaces after commas
```

**Sentence entailment:**
```bash
--text_columns s1,s2  # Processes as sentence pairs
```
