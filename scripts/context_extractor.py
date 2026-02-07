import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

import sys
import subprocess
import json
import warnings
def install_dependencies():
    required_packages = [
        ("torch", "torch"),
        ("transformers", "transformers==4.41.2"),
        ("PIL", "Pillow"),
        ("timm", "timm"),
        ("einops", "einops"),
        ("requests", "requests"),
        ("numpy", "numpy<2"),
    ]

    needs_restart = False
    for module_name, pip_name in required_packages:
        try:
            __import__(module_name)
        except ImportError:
            print(f"[Auto-Install] Installing {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "--quiet"])
            needs_restart = True

    if needs_restart:
        print("[System] Dependencies installed. Restarting script...")
        os.execv(sys.executable, ['python'] + sys.argv)
install_dependencies()
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

warnings.filterwarnings("ignore")
DEFAULT_SOURCE = "../example_resources/sample_image.jpg"


def load_model(device):
    print(f"[Init] Loading Microsoft/Florence-2-large on {device}...")
    dtype = torch.float16 if device.type == 'cuda' else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="eager"
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True
    )
    return model, processor


def load_image(image_path_or_url):
    print(f"[IO] Loading image from: {image_path_or_url}")
    try:
        if image_path_or_url.startswith("http"):
            return Image.open(requests.get(image_path_or_url, stream=True).raw).convert("RGB")
        else:
            if not os.path.exists(image_path_or_url):
                raise FileNotFoundError(f"File not found: {image_path_or_url}")
            return Image.open(image_path_or_url).convert("RGB")
    except Exception as e:
        print(f"[Error] Could not load image: {e}")
        sys.exit(1)


def run_inference(model, processor, image, task_prompt, device):
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)

    if device.type == 'cuda':
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        num_beams=1,
        use_cache=False,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    # parsed_result = processor.post_process_generation(
    #     generated_text,
    #     task=task_prompt,
    #     image_size=(image.width, image.height)
    # )
    image_size = image.size
    parsed_result = processor.post_process_generation(generated_text, task=task_prompt, image_size=image_size)

    return parsed_result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_source = sys.argv[1]
    else:
        print(f"[Info] No image provided. Using default image.")
        image_source = DEFAULT_SOURCE

    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Resources
    model, processor = load_model(device)
    image = load_image(image_source)

    print(f"[Processing] Image size: {image.size}")

    # 3. Define Tasks
    tasks = {
        "Object Detection": "<OD>",
        "OCR": "<OCR>"
    }

    final_output = {"source": image_source, "results": {}}

    # 4. Run Inference
    for task_name, task_prompt in tasks.items():
        print(f"[Inference] Running {task_name}...")
        result = run_inference(model, processor, image, task_prompt, device)
        final_output["results"][task_name] = result[task_prompt]

    print("\n" + "=" * 40)
    print("FINAL RESULTS (JSON)")
    print("=" * 40)
    print(json.dumps(final_output, indent=2))

    filename = "../example_resources/florence2_output.json"
    with open(filename, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n[Done] Results saved to {filename}")

#RUN USING: python scripts/context_extractor.py