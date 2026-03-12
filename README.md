# Fine-Tuning Exercise: Astrologer Personalities with LoRA

Fine-tune a small language model (Qwen2.5-0.5B) into two distinct astrologer personas using LoRA adapters. Same base model, same questions, completely different personalities.

**Mystic Maya** — Warm, poetic, spiritual. Speaks in metaphors and cosmic wisdom.

**Brutal Brad** — Blunt, sarcastic, funny. No sugarcoating, just truth bombs.

## Prerequisites

- A Google account (free — for Google Colab)
- A Hugging Face account (free — to download the model, sign up at [huggingface.co](https://huggingface.co))

## Step-by-Step: Getting Started with Google Colab

Google Colab is a free online tool from Google that lets you run Python code in your browser — with access to a GPU (the hardware needed for AI training). No installation required.

### 1. Open the Notebook in Google Colab

Click this link to open the notebook directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arlocinsights/fine-tuning-exercise/blob/main/fine_tuning_exercise.ipynb)

Sign in with your Google account if prompted.

**Always use this link** — it pulls the latest version from GitHub. Do not upload the notebook manually or use a saved copy.

### 2. Enable GPU (Important!)

The exercise needs a GPU to run. Colab gives you one for free:

- Click **Runtime** in the top menu
- Click **Change runtime type**
- Under **Hardware accelerator**, select **T4 GPU**
- Click **Save**

### 3. Run the Notebook

The notebook is made up of **cells** — blocks of code or text. To run a cell:

- Click on the cell to select it
- Press **Shift + Enter** to run it (or click the play button on the left)
- Wait for it to finish (you'll see a green checkmark) before running the next one
- Run the cells **from top to bottom** — each cell depends on the ones before it

The first code cell will clone this repository and install all the required packages. This takes about 1-2 minutes.

### 4. Fill in the TODOs

As you go through the notebook, you'll see three places where you need to fill in a value (marked with `___`):

1. **Model name** (Step 1) — replace `"___"` with the Hugging Face model ID
2. **LoRA rank** (Step 5) — replace `___` with a number (start with `8`)
3. **Training epochs** (Steps 6 & 7) — replace `___` with a number (start with `3`)

To edit a code cell, just click on it and type.

### 5. Edit the Training Data

You'll also add 3 more training examples to each persona's data file:

- Click the **folder icon** on the left sidebar to open the file browser
- Navigate to `fine-tuning-exercise/training_data/maya/` and double-click `maya_training.json`
- Add your examples to the JSON file following the existing pattern
- Do the same for `fine-tuning-exercise/training_data/brad/brad_training.json`
- **Save** each file (Ctrl+S / Cmd+S) before running the training cells

### Troubleshooting

| Problem | Solution |
|---------|----------|
| "Runtime disconnected" | Click **Reconnect** in the top right. Colab disconnects after ~90 min of inactivity. |
| "Out of memory" | Click **Runtime > Restart runtime**, then re-run from the top. |
| Cell seems stuck | Check for a spinning icon. Some cells (training) take a few minutes. Be patient! |
| "Module not found" error | Make sure you ran the first cell (Step 0) that installs dependencies. |
| Can't find the files to edit | Click the folder icon on the left sidebar. If empty, re-run the first cell. |

## Project Structure

```
├── fine_tuning_exercise.ipynb       # Main notebook — run this in Colab
├── requirements.txt                 # Python dependencies
├── README.md
├── training_data/
│   ├── maya/
│   │   └── maya_training.json       # Mystic Maya's training examples
│   └── brad/
│       └── brad_training.json       # Brutal Brad's training examples
```

## What You'll Do

| Step | What happens |
|------|-------------|
| 0 | Clone repo and install dependencies in Colab |
| 1 | Load the base model and tokenizer |
| 2 | Ask the base model astrology questions — notice the generic answers |
| 3 | Load personality-specific training data from JSON files |
| 4 | Format the data into the chat template the model expects |
| 5 | Configure LoRA (you choose the rank) |
| 6 | Train Mystic Maya's adapter |
| 7 | Train Brutal Brad's adapter |
| 8 | Compare all three side-by-side on unseen questions |
| 9 | Check how small the adapters are compared to the full model |

## TODOs in the Notebook

There are three values you need to fill in (marked with `___`):

1. **Model name** (Step 1) — the Hugging Face model ID
2. **LoRA rank** (Step 5) — start with `8`
3. **Training epochs** (Steps 6 & 7) — start with `3`

You also need to add **3 more training examples** to each persona's JSON file.

## Adding Training Data

Edit the JSON files in `training_data/` (use Colab's file browser on the left sidebar). Each example is a question-answer pair:

```json
{
    "question": "Your astrology question here",
    "answer": "Response in the persona's voice and style"
}
```

Keep the personality consistent! Maya should always sound spiritual and poetic. Brad should always sound blunt and sarcastic.

## Things to Experiment With

### LoRA Rank (`r`)
Controls how many trainable parameters the adapter has.

| Rank | Parameters | Effect |
|------|-----------|--------|
| 4 | Fewer | Faster training, may lose personality nuance |
| 8 | Moderate | Good starting point |
| 16 | More | Stronger personality, slower training |
| 32 | Many | Diminishing returns, risk of overfitting |

### Training Epochs
How many times the model sees the full dataset.

- **1-2 epochs**: Personality is faint, model still sounds generic
- **3-5 epochs**: Sweet spot — personality is clear without overfitting
- **10+ epochs**: Model starts memorizing training data word-for-word (bad!)

### Dataset Size
Start with the provided 7 examples per persona, then try adding more.

- **7-10 examples**: Personality comes through but can be inconsistent
- **20-30 examples**: More reliable personality on unseen questions
- **50+ examples**: Diminishing returns for this exercise

### Target Modules
The default config applies LoRA to `q_proj` and `v_proj` (attention layers). Try adding more:

```python
target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
```

More modules = more trainable parameters = potentially stronger personality shift.

## Bonus Challenges

1. **Create a third persona** — a dramatic soap-opera astrologer, a Gen-Z astrologer who uses slang, or a skeptical scientist who reluctantly gives readings. Create a new folder in `training_data/` and add a training section in the notebook.

2. **Compare adapter sizes** — after training with different ranks, check how the adapter file sizes change. Is 2x the rank worth 2x the storage?

3. **Test generalization** — ask questions about signs that aren't in the training data. Does the personality still come through?

4. **Break it on purpose** — train for 50 epochs and see what overfitting looks like. Compare the outputs to the 3-epoch version.

## Key Concepts

- **Fine-tuning** adapts a pretrained model to behave differently (new style, new personality, new domain)
- **LoRA** makes fine-tuning practical by only training a tiny fraction of the model's parameters
- **Adapters** are small files (a few MB) that sit on top of a large base model — you can swap them at inference time
- This is how real apps serve multiple personalities/styles from a single model deployment
