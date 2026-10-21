import os
import warnings
from pathlib import Path

# Suppress TensorFlow logging and non-critical Keras warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

import tensorflow as tf
from keras.src.layers.preprocessing.index_lookup import IndexLookup

# Defensive patch for Keras 3 TextVectorization / IndexLookup bug:
# Keras uses f.read().splitlines() which breaks on Unicode linebreaks like \x85 (NEL)
# causing duplicate empty tokens and a ValueError during model loading.
_orig_load_assets = IndexLookup.load_assets

def _safe_load_assets(self, dir_path):
    if self.input_vocabulary is not None:
        return
    vocabulary_filepath = tf.io.gfile.join(dir_path, "vocabulary.txt")
    with open(vocabulary_filepath, "r", encoding="utf-8") as f:
        # Split strictly on '\n' rather than all Unicode newline characters
        lines = f.read().split("\n")
        while lines and lines[-1] == "":
            lines.pop()
        if tf.as_dtype(self.vocabulary_dtype) == tf.string:
            values = [str(line) for line in lines]
        else:
            values = [int(line) for line in lines]
        if self.output_mode == "tf_idf":
            idf_weights = self.idf_weights_const.numpy()
            self.set_vocabulary(values, idf_weights=idf_weights)
        else:
            self.set_vocabulary(values)

IndexLookup.load_assets = _safe_load_assets

# Resolve model path relative to this script directory
MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.keras"

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Get review input from user
user_input = input("Enter your review: ").strip()
while not user_input:
    print("Review cannot be empty. Please enter a valid review.")
    user_input = input("Enter your review: ").strip()

to_review = tf.constant([user_input])

# Run prediction (verbose=0 suppresses the progress bar output)
output = model.predict(to_review, verbose=0)

# Extract scalar probability: model outputs sigmoid probability for class 1 (Positive)
positive_prob = float(output[0][0])

# Determine sentiment and calculate correct confidence percentage
if positive_prob >= 0.5:
    review = "Positive"
    confidence = positive_prob * 100
else:
    review = "Negative"
    confidence = (1.0 - positive_prob) * 100

print(f"\nModel Prediction: {review}\nConfidence: {confidence:.2f}%")


"""
================================================================================
FIXED BUGS AND REFERENCE NOTES
================================================================================

1. Model Loading Failure (ValueError: repeated term [np.str_('')]):
   - Cause: The trained model's TextVectorization layer contains a Unicode next-line
     character (\\x85 / U+0085) in its vocabulary from the IMDb dataset. When Keras 3
     loads vocabulary.txt using Python's `read().splitlines()`, Python treats \\x85 as
     a newline boundary. This splits the single token into two empty strings, causing
     Keras to detect duplicate '' tokens and raise a loading error.
   - Fix:
     a) Repaired `model/model.keras` by replacing the breaking character in `vocabulary.txt`
        (with a backup preserved at `model/model.keras.bak`).
     b) Added a defensive patch to `IndexLookup.load_assets` in this file to split strictly
        on standard '\\n', preventing future loading failures even if unpatched models are used.

2. Inverted Surety/Confidence Calculation for Negative Predictions:
   - Cause: In binary classification with a sigmoid output:
       Output value = Probability of being POSITIVE (P(Positive) in range [0, 1]).
     When output was < 0.5 (e.g., 0.05 -> Negative), multiplying `output * 100` reported
     "Negative with 5% surety". In reality, the model was 95% confident that the review was
     Negative (1.0 - 0.05 = 0.95).
   - Fix: For Positive predictions (prob >= 0.5), confidence is `prob * 100`.
     For Negative predictions (prob < 0.5), confidence is `(1.0 - prob) * 100`.

3. Multi-dimensional Array Comparison and Scalar Extraction:
   - Cause: `model.predict(to_review)` returns a 2D NumPy array of shape (1, 1), e.g., `[[0.27]]`.
     Evaluating `if output < .5:` directly on an array can raise `ValueError: The truth value of an
     array with more than one element is ambiguous` in newer NumPy versions or batch contexts.
   - Fix: Extracted the scalar float using `float(output[0][0])`.

4. Output Formatting (Raw Array Brackets in String Interpolation):
   - Cause: In `f"with {output*100}% surety"`, printing the array directly resulted in raw
     NumPy bracket representations like `[[27.065823]]% surety`.
   - Fix: Formatted the extracted scalar float with two decimal places (`{confidence:.2f}%`).

5. Progress Bar Clutter and Warning Noise:
   - Cause: Calling `model.predict()` in interactive CLI scripts defaults to printing a
     progress bar (e.g., `1/1 ━━━━━━━━━━━━━━━━━━━━ 0s`), and Keras/TensorFlow print mask warnings
     and verbose C++ logging.
   - Fix: Set `verbose=0` on `model.predict()`, set `TF_CPP_MIN_LOG_LEVEL = '2'`, and filtered
     non-critical Keras user warnings.

6. Fragile Relative Path:
   - Cause: `'model/model.keras'` depends on running Python from the project root directory. Running
     the script from another working directory throws `FileNotFoundError`.
   - Fix: Used `Path(__file__).resolve().parent / "model" / "model.keras"` for location-independent
     execution.

7. Empty / Whitespace Input Handling:
   - Cause: Entering blank/whitespace input directly caused an unvalidated tensor to be evaluated.
   - Fix: Added an input validation loop with `.strip()` to ensure a valid review string is provided.
================================================================================
"""
