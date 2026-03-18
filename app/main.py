import gradio as gr
import io
import random
from pathlib import Path

from PIL import Image
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from app.model import CLASS_NAMES, gradcam, predict

HF_DATASET_REPO = "AIOmarRehan/COVID-19"
_hf_api = HfApi()
_dataset_image_files: list[str] | None = None
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _extract_image_paths(paths: list[str]) -> list[str]:
    return [file_path for file_path in paths if Path(file_path).suffix.lower() in IMAGE_EXTENSIONS]


def _list_from_repo_tree() -> list[str]:
    tree_entries = _hf_api.list_repo_tree(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        recursive=True,
    )
    repo_paths = [entry.path for entry in tree_entries if hasattr(entry, "path")]
    return _extract_image_paths(repo_paths)


def _list_from_snapshot() -> list[str]:
    snapshot_dir = snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        allow_patterns=[
            "**/*.png",
            "**/*.jpg",
            "**/*.jpeg",
            "**/*.bmp",
            "**/*.webp",
            "**/*.tif",
            "**/*.tiff",
        ],
    )
    return [
        str(path)
        for path in Path(snapshot_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _get_dataset_image_files() -> list[str]:
    global _dataset_image_files
    if _dataset_image_files is None:
        files = _hf_api.list_repo_files(repo_id=HF_DATASET_REPO, repo_type="dataset")
        image_files = _extract_image_paths(files)

        if not image_files:
            image_files = _list_from_repo_tree()

        if not image_files:
            image_files = _list_from_snapshot()

        _dataset_image_files = image_files
    return _dataset_image_files


def _load_random_image_from_dataset_rows():
    from datasets import load_dataset

    try:
        dataset_dict = load_dataset(HF_DATASET_REPO)
        split_name = "train" if "train" in dataset_dict else next(iter(dataset_dict.keys()))
        dataset_split = dataset_dict[split_name]
    except Exception:
        dataset_split = load_dataset(HF_DATASET_REPO, split="train")
        split_name = "train"

    if len(dataset_split) == 0:
        raise ValueError("Dataset split is empty")

    sample_index = random.randint(0, len(dataset_split) - 1)
    sample = dataset_split[sample_index]

    for column_name, value in sample.items():
        if isinstance(value, Image.Image):
            return value.convert("RGB"), f"Loaded from {HF_DATASET_REPO}: split={split_name}, column={column_name}, index={sample_index}"

        if isinstance(value, dict):
            if value.get("bytes"):
                image_obj = Image.open(io.BytesIO(value["bytes"]))
                return image_obj.convert("RGB"), f"Loaded from {HF_DATASET_REPO}: split={split_name}, column={column_name}, index={sample_index}"
            if value.get("path"):
                image_obj = Image.open(value["path"])
                return image_obj.convert("RGB"), f"Loaded from {HF_DATASET_REPO}: split={split_name}, column={column_name}, index={sample_index}"

        if isinstance(value, str) and Path(value).suffix.lower() in IMAGE_EXTENSIONS:
            image_obj = Image.open(value)
            return image_obj.convert("RGB"), f"Loaded from {HF_DATASET_REPO}: split={split_name}, column={column_name}, index={sample_index}"

    raise ValueError("No image-like column found in sampled dataset row")


def run_prediction(image: Image.Image):
    if image is None:
        return "No image provided", 0.0, {name: 0.0 for name in CLASS_NAMES}

    label, confidence, probabilities = predict(image)
    return label, confidence, probabilities


def run_gradcam(image: Image.Image, interpolant: float):
    if image is None:
        return None
    return gradcam(image, interpolant=interpolant)


def load_random_dataset_image():
    try:
        image_files = _get_dataset_image_files()
        if not image_files:
            return _load_random_image_from_dataset_rows()

        selected_file = random.choice(image_files)
        if Path(selected_file).exists():
            local_path = selected_file
        else:
            local_path = hf_hub_download(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                filename=selected_file,
            )
        random_image = Image.open(local_path).convert("RGB")
        return random_image, f"Loaded from {HF_DATASET_REPO}: {selected_file} (found {len(image_files)} images)"
    except Exception as exc:
        return None, f"Could not load random image from {HF_DATASET_REPO}. Error: {exc}"


with gr.Blocks(title="COVID 19 Radiography Classification") as demo:
    gr.Markdown("# COVID 19 Radiography Classification")
    gr.Markdown("Upload a chest X ray image to run classification and Grad CAM visualization.")

    with gr.Row():
        input_image = gr.Image(type="pil", label="Input X ray")
        gradcam_output = gr.Image(type="numpy", label="Grad CAM")

    with gr.Row():
        predicted_label = gr.Textbox(label="Predicted Class")
        confidence = gr.Number(label="Confidence")

    probability_output = gr.Label(label="Class Probabilities", num_top_classes=len(CLASS_NAMES))
    dataset_image_info = gr.Textbox(label="Dataset Image Source", interactive=False)
    interpolant = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Grad CAM Interpolant")

    with gr.Row():
        random_image_button = gr.Button("Random Dataset Image")
        predict_button = gr.Button("Predict")
        gradcam_button = gr.Button("Generate Grad CAM")
        all_button = gr.Button("Run All")

    random_image_button.click(
        fn=load_random_dataset_image,
        inputs=None,
        outputs=[input_image, dataset_image_info],
    )

    predict_button.click(
        fn=run_prediction,
        inputs=input_image,
        outputs=[predicted_label, confidence, probability_output],
    )

    gradcam_button.click(
        fn=run_gradcam,
        inputs=[input_image, interpolant],
        outputs=gradcam_output,
    )

    all_button.click(
        fn=run_prediction,
        inputs=input_image,
        outputs=[predicted_label, confidence, probability_output],
    ).then(
        fn=run_gradcam,
        inputs=[input_image, interpolant],
        outputs=gradcam_output,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)