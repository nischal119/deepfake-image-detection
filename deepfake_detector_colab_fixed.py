import gc
import os
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import time

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import ClassLabel, Dataset, Image
from imblearn.over_sampling import RandomOverSampler
from PIL import ImageFile
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomAdjustSharpness,
    RandomRotation,
    Resize,
    ToTensor,
)
from tqdm import tqdm
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def in_colab() -> bool:
    return "google.colab" in sys.modules


def ensure_kaggle_dataset_colab(data_dir: Path) -> Path:
    """
    In Colab: download and extract the dataset into data_dir.
    Returns the extracted dataset root (data_dir / "Dataset").
    """
    zip_path = data_dir / "deepfake-and-real-images.zip"
    extracted_dir = data_dir / "Dataset"

    data_dir.mkdir(parents=True, exist_ok=True)

    # Install kaggle if missing
    try:
        subprocess.run(
            ["kaggle", "-h"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "kaggle"]
        )  # quiet install

    # Prepare kaggle creds
    kaggle_json_src = Path("/content/kaggle.json")
    kaggle_dir = Path("/root/.kaggle")
    if kaggle_json_src.exists():
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            ["cp", str(kaggle_json_src), str(kaggle_dir / "kaggle.json")]
        )
        subprocess.check_call(["chmod", "600", str(kaggle_dir / "kaggle.json")])
    else:
        print(
            "kaggle.json not found at /content/kaggle.json. Skipping download; expecting dataset to be present."
        )

    if not zip_path.exists() and kaggle_json_src.exists():
        print("Downloading dataset from Kaggle...")
        subprocess.check_call(
            [
                "kaggle",
                "datasets",
                "download",
                "manjilkarki/deepfake-and-real-images",
                "-p",
                str(data_dir),
                "--force",
            ]
        )
    else:
        print("Dataset zip already exists or kaggle.json missing; skipping download.")

    if not extracted_dir.exists() and zip_path.exists():
        print("Extracting dataset zip...")
        subprocess.check_call(["unzip", "-q", str(zip_path), "-d", str(data_dir)])
        print("Extraction completed.")
    else:
        print("Dataset already extracted or zip missing.")

    return extracted_dir


def resolve_dataset_root() -> Path:

    if in_colab():
        base = ensure_kaggle_dataset_colab(Path("/content/deepfake_dataset"))
        return base
    # Local/non-Colab
    local = Path("Dataset")
    if not local.exists():
        raise FileNotFoundError(
            f"Dataset folder not found at {local.resolve()}. Place your dataset there (Train/Real, Train/Fake, etc.)."
        )
    return local


def build_dataframe(base_path: Path) -> pd.DataFrame:
    print("Scanning dataset directory...")
    file_names, labels = [], []

    # Get all jpg files first
    all_files = list(base_path.rglob("*.jpg"))
    print(f"Found {len(all_files)} image files")

    # Process with progress bar
    for file in tqdm(all_files, desc="📊 Building dataset", unit="images"):
        labels.append(file.parent.name)
        file_names.append(str(file))

    df = pd.DataFrame({"image": file_names, "label": labels})
    print(f"Dataset created with {len(df)} images")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    return df


def main():
    print("Starting DeepFake Detector Training Pipeline")
    print("=" * 50)

    # Step 1: Dataset setup
    print("\nStep 1: Setting up dataset...")
    dataset_root = resolve_dataset_root()
    print(f"📍 Using dataset at: {dataset_root.resolve()}")

    # Step 2: Build dataframe
    print("\nStep 2: Building dataset dataframe...")
    df = build_dataframe(dataset_root)

    # Step 3: Balance dataset
    print("\nStep 3: Balancing dataset...")
    print("Applying RandomOverSampler...")
    y = df[["label"]]
    df_x = df.drop(["label"], axis=1)
    ros = RandomOverSampler(random_state=83)
    df_x, y_resampled = ros.fit_resample(df_x, y)
    df = df_x.copy()
    df["label"] = y_resampled
    del y_resampled, df_x
    gc.collect()
    print(f"✅ Dataset balanced: {df.shape[0]} samples")
    print(f"📈 Balanced distribution:\n{df['label'].value_counts()}")

    # Step 4: Create HuggingFace dataset
    print("\n🤗 Step 4: Creating HuggingFace dataset...")
    with tqdm(total=3, desc="🔄 Processing dataset") as pbar:
        dataset = Dataset.from_pandas(df).cast_column("image", Image())
        pbar.update(1)

        labels_list = ["Real", "Fake"]
        label2id = {name: idx for idx, name in enumerate(labels_list)}
        id2label = {idx: name for name, idx in label2id.items()}
        pbar.update(1)

        class_labels = ClassLabel(num_classes=len(labels_list), names=labels_list)

        def map_label2id(example):
            example["label"] = class_labels.str2int(example["label"])
            return example

        dataset = dataset.map(map_label2id, batched=True)
        dataset = dataset.cast_column("label", class_labels)
        pbar.update(1)

    print("✅ HuggingFace dataset created")

    # Step 5: Split dataset
    print("\n✂️  Step 5: Splitting dataset...")
    print("🔄 Creating train/test split (60/40)...")
    dataset = dataset.train_test_split(
        test_size=0.4, shuffle=True, stratify_by_column="label"
    )
    train_data = dataset["train"]
    test_data = dataset["test"]

    print(f"✅ Split complete:")
    print(f"   📚 Training samples: {len(train_data)}")
    print(f"   🧪 Test samples: {len(test_data)}")

    # Step 6: Load model and processor
    print("\n🤖 Step 6: Loading model and processor...")
    model_str = "dima806/deepfake_vs_real_image_detection"
    print("🔄 Loading ViT processor...")
    processor = ViTImageProcessor.from_pretrained(model_str)

    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]
    print(f"📏 Image size: {size}")

    # Step 7: Setup transforms
    print("\n🔄 Step 7: Setting up image transforms...")
    normalize = Normalize(mean=image_mean, std=image_std)

    _train_transforms = Compose(
        [
            Resize((size, size)),
            RandomRotation(90),
            RandomAdjustSharpness(2),
            ToTensor(),
            normalize,
        ]
    )

    _val_transforms = Compose(
        [
            Resize((size, size)),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(examples):
        if "image" not in examples:
            return examples
        examples["pixel_values"] = [
            _train_transforms(image.convert("RGB")) for image in examples["image"]
        ]
        return examples

    def val_transforms(examples):
        if "image" not in examples:
            return examples
        examples["pixel_values"] = [
            _val_transforms(image.convert("RGB")) for image in examples["image"]
        ]
        return examples

    train_data.set_transform(train_transforms)
    test_data.set_transform(val_transforms)
    print("✅ Transforms applied")

    # Step 8: Setup model and training
    print("\n🏗️  Step 8: Setting up model and training...")

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    print("🔄 Loading ViT model...")
    model = ViTForImageClassification.from_pretrained(
        model_str,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )
    print(
        f"✅ Model loaded: {model.num_parameters(only_trainable=True)/1e6:.1f}M parameters"
    )

    print("🔄 Setting up metrics...")
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        label_ids = eval_pred.label_ids
        preds = predictions.argmax(axis=1)
        return {
            "accuracy": metric.compute(predictions=preds, references=label_ids)[
                "accuracy"
            ]
        }

    print("🔄 Configuring training arguments...")
    args = TrainingArguments(
        output_dir="deepfake_vs_real_image_detection",
        logging_dir="./logs",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        learning_rate=1.6e-4,
        weight_decay=0.02,
        warmup_steps=50,
        save_total_limit=1,
        do_train=True,
        do_eval=True,
        report_to="none",
        logging_steps=100,
        remove_unused_columns=False,
    )

    print("🔄 Creating trainer...")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    print("✅ Trainer ready")

    # Step 9: Training
    print("\n🎯 Step 9: Starting training...")
    print("🔥 Training for 2 epochs...")
    trainer.train()
    print("✅ Training completed!")

    # Step 10: Evaluation
    print("\n📊 Step 10: Evaluating model...")
    results = trainer.evaluate()
    print("🎉 Final Results:")
    for key, value in results.items():
        print(f"   {key}: {value:.4f}")


if __name__ == "__main__":
    main()
