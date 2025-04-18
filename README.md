# DL_Assignment_2



# Question-2
## 🎵 GPT-2 Lyrics Generator

This project fine-tunes a pre-trained GPT-2 language model on a small set of custom lyrics and generates new lyrics based on a user prompt. It uses the 🤗 Hugging Face Transformers and Datasets libraries.

---

## 💡 Project Overview

This simple project demonstrates how to:

- Use a small custom dataset of lyrics.
- Tokenize and prepare text data for GPT-2.
- Fine-tune GPT-2 for language modeling.
- Generate new lyrics based on a user-defined prompt.

---

## 🛠️ Installation

Before running the project, install the necessary dependencies:

```bash
pip install datasets transformers torch
```

---

## 📁 Project Structure

```
.
├── lyrics.txt             # Sample lyrics file
├── gpt2-lyrics-model/     # Saved model after fine-tuning
├── gpt2-lyrics-output/    # Output directory during training
├── lyrics_generator.py    # Python script with training and generation code
└── README.md              # This file
```

---

## 🚀 Training the Model

The script fine-tunes the GPT-2 model on a small set of example lyrics (`lyrics.txt`). If the file doesn't exist, it will be created with sample lines.

### Steps:

1. Load and tokenize the lyrics dataset.
2. Use `DataCollatorForLanguageModeling` to prepare batches.
3. Fine-tune the GPT-2 model for 3 epochs using `Trainer`.
4. Save the fine-tuned model and tokenizer.

You can find the trained model in the `gpt2-lyrics-model/` directory.

---

## 🎤 Generating Lyrics

After training, the script asks the user for a prompt and generates a continuation of the lyrics.

### Example:

```bash
Enter your song prompt: You are the light in my life
```

**Generated Output:**
```
You are the light in my life
Guiding me through the darkest night
With every step I take, I feel you
And everything suddenly feels right
```

---

## ⚙️ Configuration Details

- **Model**: GPT-2 from Hugging Face
- **Max Token Length**: 128
- **Batch Size**: 2
- **Training Epochs**: 3
- **Text Generation Settings**:
  - `temperature=0.7`
  - `top_k=40`
  - `top_p=0.95`
  - `do_sample=True`

---

## 📌 Notes

- This example uses a very small dataset; for better results, train on a larger and more diverse lyrics corpus.
- WANDB logging is disabled by setting `os.environ["WANDB_DISABLED"] = "true"`.

---

## ✨ Future Improvements

- Use a larger and genre-specific lyrics dataset.
- Add a web interface for interactive generation.
- Include rhyme and rhythm constraints during generation.



