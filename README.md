# Prompt Injection Detector

A machine learning classifier and pre-processing pipeline built to detect prompt injection attacks targeting Large Language Models (LLMs). This project features a **pre-trained model binary** saved directly in the repository for instant inference, alongside training and evaluation logic.

---

## Project Features
- **Saved Model Binary:** Pre-trained `TF-IDF + Logistic Regression` pipeline serialized with `joblib` (`models/pipeline.joblib`).
- **Defense & Normalization Layer:** Pre-processing logic to neutralize token-evasion tricks (hyphen-splitting) and a heuristic decoder to detect hidden Base64 payloads.
- **Adversarial Testing:** Evaluates the model against rephrased, encoded, and formatted injection attempts to measure robustness drops.

---

## Project Structure
text
prompt-injection-detector/
│
├── data/
│   └── dataset.csv          # Training data
├── models/
│   └── pipeline.joblib      # Pre-trained model binary (ready on Git)
├── train.py                 # Script to train and save/export the model
├── predict.py               # Inference script using the saved model binary
├── requirements.txt         # Dependencies
└── README.md                # Documentation

## Getting Started
1. Clone the Repository
Bash
git clone [https://github.com/AvilashDakour/prompt-injection-detector.git](https://github.com/AvilashDakour/prompt-injection-detector.git)
cd prompt-injection-detector
2. Set Up Virtual Environment & Install Dependencies
Bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
3. Run Inference with the Pre-trained Model
You can directly classify prompts using the trained binary shipped with this repo:
python predict.py

4. Retrain the Model (Optional)
If you update data/dataset.csv with new samples, re-run training to export an updated model file:
python train.py

Key Takeaways
Binary Deployment: Serializing trained model pipelines (.joblib) allows instant deployment without retraining during inference.


Adversarial Vulnerability: Simple statistical models suffer a robustness drop on semantic variations, showing why production guardrails combine classifiers with semantic embeddings or LLM-as-a-judge patterns.

License
This project is open-source and available under the MIT License.