# Model card

## What it does

Scores Russian Discord message text from 1 (benign) to 10 (severe). The bot uses this score for configurable warning and timeout actions.

## How it was built

The included artifact contains a TF-IDF character/word feature pipeline with a Ridge regressor. It was exported from the original project with the raw feedback text and score arrays removed. The source artifact reports 5,000 teacher-labelled examples and 483 feedback examples; the model weights still reflect those examples.

## Limits and responsible use

This is a baseline, not a general-purpose safety classifier. It can misunderstand sarcasm, context, slang, and reclaimed language. Evaluate on data you have permission to use, tune thresholds for your community, and provide a human review path. Before redistributing the included weights publicly, confirm that the training data and feedback-derived model are appropriate to share.

Only load model artifacts from trusted sources: `joblib` files use Python pickle serialization.
