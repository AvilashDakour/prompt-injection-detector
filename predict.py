import base64
import re
import joblib
import sys

# 1. Load the pre-trained model from disk
model_path = "models/pipeline.joblib"
try:
    pipeline = joblib.load(model_path)
    print(f"Loaded pre-trained model from {model_path}\n")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please run `python train.py` first to generate the model file.")
    sys.exit(1)

# 2. Defense Layer & Classification logic
def predict_prompt(prompt):
    # Base64 check
    base64_pattern = re.compile(r'([A-Za-z0-9+/]{20,}={0,2})')
    matches = base64_pattern.findall(prompt)
    for match in matches:
        try:
            decoded_bytes = base64.b64decode(match)
            decoded_str = decoded_bytes.decode('utf-8', errors='ignore')
            if any(kw in decoded_str.lower() for kw in ["ignore", "disregard", "override", "system"]):
                return "BLOCKED (Hidden Base64 Attack Detected)", 0.99
        except Exception:
            pass

    # Normalization check
    cleaned_prompt = re.sub(r'(?<=\w)-(?=\w)', '', prompt)
    cleaned_prompt = re.sub(r'\s+', ' ', cleaned_prompt).lower()

    # Predict using saved pipeline
    pred = pipeline.predict([cleaned_prompt])[0]
    prob = pipeline.predict_proba([cleaned_prompt])[0][1]

    status = "INJECTION DETECTED" if pred == 1 else "✅ Benign"
    return status, prob

# 3. Interactive Custom Prompt Loop
if __name__ == "__main__":
    print("Prompt Injection Detector - Interactive Mode")
    print("Type any prompt to test it against your model. Type 'exit' or 'quit' to close.\n" + "-"*50)
    
    while True:
        try:
            user_prompt = input("\nEnter prompt to test: ").strip()
            
            if user_prompt.lower() in ["exit", "quit"]:
                print("Exiting detector. Goodbye!")
                break
                
            if not user_prompt:
                continue
            
            status, prob = predict_prompt(user_prompt)
            print(f"Result: {status} (Injection Probability: {prob:.2f})")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nExiting detector. Goodbye!")
            break