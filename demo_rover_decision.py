import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

# ----- CONFIG -----
CLEAN_DIR = Path("images/clean")
DIRTY_DIR = Path("images/dirty")
NUM_PER_CLASS = 5  # 5 clean + 5 dirty

# Set for reproducible demos (optional)
random.seed(42)


def get_image_paths(directory: Path, max_images: int):
    """Return up to max_images image paths from a folder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = [p for p in directory.iterdir() if p.suffix.lower() in exts]
    if not paths:
        raise FileNotFoundError(f"No images found in {directory}")
    random.shuffle(paths)
    return paths[:max_images]


# ---------- MODEL PLACEHOLDER / HOOK ----------

def load_rover_model():
    """
    TODO: Replace this with your real model loader.
    e.g., YOLO model, PyTorch, TensorFlow, etc.
    """
    model = None
    return model


def classify_with_model(model, img_path: Path) -> str:
    """
    Return 'clean' or 'dirty' for the given image.

    Right now this is a dummy function that infers the label from the filename.
    Replace the middle section with a call to your real model.
    """
    # ---- DUMMY LOGIC (works even without ML model) ----
    name = img_path.name.lower()
    if "dirty" in name or "trash" in name or "plastic" in name:
        return "dirty"
    else:
        return "clean"

    # ---- REAL LOGIC EXAMPLE (pseudo-code) ----
    # if model is not None:
    #     img = cv2.imread(str(img_path))
    #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     # Preprocess according to your model
    #     pred = model.predict(img_rgb)
    #     return "clean" if pred == 0 else "dirty"


# ---------- VISUALIZATION HELPERS ----------

def show_gallery(clean_paths, dirty_paths):
    """
    Show 5 clean and 5 dirty images in a 2x5 grid.
    Top row: clean, bottom row: dirty.
    """
    fig, axes = plt.subplots(2, NUM_PER_CLASS, figsize=(14, 5))
    fig.suptitle("Demo Set: Clean vs Dirty Regolith Views", fontsize=16)

    # Ensure axes is 2D, even if NUM_PER_CLASS=1
    if NUM_PER_CLASS == 1:
        axes = axes.reshape(2, 1)

    # Top row: clean
    for i, img_path in enumerate(clean_paths):
        ax = axes[0, i]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(f"Clean #{i+1}", fontsize=8)
        ax.axis("off")

    # Bottom row: dirty
    for i, img_path in enumerate(dirty_paths):
        ax = axes[1, i]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(f"Dirty #{i+1}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def show_single_decision(img_path: Path, true_label: str, predicted_label: str, action: str):
    """
    Show one image and overlay the rover's prediction + action.
    """
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title(
        f"Rover View: {img_path.name}\n"
        f"True: {true_label.upper()} | Predicted: {predicted_label.upper()} | Action: {action}",
        fontsize=10
    )
    plt.tight_layout()
    plt.show()


# ---------- MAIN DEMO LOGIC ----------

def main():
    # 1. Load demo images
    clean_paths = get_image_paths(CLEAN_DIR, NUM_PER_CLASS)
    dirty_paths = get_image_paths(DIRTY_DIR, NUM_PER_CLASS)

    # 2. Show gallery of 5 clean + 5 dirty
    show_gallery(clean_paths, dirty_paths)

    # 3. Load model (real or placeholder)
    model = load_rover_model()

    # 4. Randomly pick one of the 10 demo images
    all_paths = clean_paths + dirty_paths
    all_true_labels = ["clean"] * len(clean_paths) + ["dirty"] * len(dirty_paths)

    idx = random.randrange(len(all_paths))
    test_path = all_paths[idx]
    true_label = all_true_labels[idx]

    # 5. Get prediction from rover brain
    predicted_label = classify_with_model(model, test_path)

    # 6. Map prediction to rover action
    if predicted_label == "clean":
        action = "SCOOP REGOLITH"
    else:
        action = "BYPASS / DO NOT SCOOP"

    # 7. Console output for narration
    print("========== ROVER DECISION DEMO ==========")
    print(f"Selected image: {test_path}")
    print(f"Ground truth label: {true_label}")
    print(f"Model prediction:  {predicted_label}")
    print(f"Rover action:      {action}")
    print("=========================================")

    # 8. Visualize the chosen image and decision
    show_single_decision(test_path, true_label, predicted_label, action)


if __name__ == "__main__":
    main()
