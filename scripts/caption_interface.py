# python
import os
import sys
import subprocess
import time
import json
import platform
import shutil
import base64

# --- Configuration ---
MODEL_NAME = "ministral-3:8b"
JSON_SOURCE = "../example_resources/florence2_output.json"
IMAGE_SOURCE = "../example_resources/sample_image.jpg"
OLLAMA_API_URL = "http://localhost:11434"


# --- 1. Robust Python Dependency Installer ---
def install_python_deps():
    """Checks for required python packages and installs them if missing."""
    required = ["requests"]
    installed = False
    for package in required:
        try:
            __import__(package)
        except ImportError:
            print(f"📦 [Auto-Install] Installing Python package: {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
            installed = True

    if installed:
        print("🔄 Dependencies installed. Restarting script...")
        os.execv(sys.executable, ['python'] + sys.argv)


install_python_deps()
import requests

def is_ollama_installed():
    return shutil.which("ollama") is not None

def install_ollama():
    system = platform.system().lower()
    if system in ["linux", "darwin"]:
        print("⚙️ [System] Installing Ollama...")
        try:
            subprocess.run(
                ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Ollama: {e}")
            sys.exit(1)
    else:
        print("❌ Automatic Ollama install not supported on Windows/Other.")
        print("➡️ Please install manually from https://ollama.com")
        sys.exit(1)


def setup_nvidia_env():
    nvidia_paths = [
        "/usr/local/nvidia/lib",
        "/usr/local/nvidia/lib64",
    ]
    existing = [p for p in nvidia_paths if os.path.isdir(p)]
    if not existing:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    new_paths = ":".join(existing)
    if new_paths not in current:
        os.environ["LD_LIBRARY_PATH"] = (
            f"{new_paths}:{current}" if current else new_paths
        )


def wait_for_server(retries=10, delay=2):
    for i in range(retries):
        try:
            response = requests.get(f"{OLLAMA_API_URL}/api/tags")
            if response.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        print(f"   ... waiting for Ollama API (attempt {i + 1}/{retries})")
        time.sleep(delay)
    return False


def ensure_ollama_running():
    if not is_ollama_installed():
        install_ollama()

    setup_nvidia_env()

    if wait_for_server(retries=1, delay=0.5):
        print("✅ [Ollama] Server is already running.")
        return

    print("🚀 [Ollama] Starting server in background...")
    try:
        with open(os.devnull, 'w') as devnull:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=devnull,
                stderr=devnull,
                preexec_fn=os.setsid if platform.system() != "Windows" else None
            )
    except Exception as e:
        print(f"❌ Failed to start Ollama server: {e}")
        sys.exit(1)

    if not wait_for_server():
        print("❌ Could not connect to Ollama server after starting.")
        sys.exit(1)
    print("✅ [Ollama] Server is ready.")


def ensure_model_pulled():
    print(f"🔍 [Model] Checking for {MODEL_NAME}...")
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags")
        models = [m['name'] for m in response.json().get('models', [])]

        if any(MODEL_NAME in m for m in models):
            print(f"✅ [Model] {MODEL_NAME} is ready.")
            return

        print(f"📦 [Model] Pulling {MODEL_NAME} (this may take a while)...")
        subprocess.run(["ollama", "pull", MODEL_NAME], check=True)
        print("✅ [Model] Pull complete.")

    except Exception as e:
        print(f"❌ Error managing models: {e}")
        sys.exit(1)


def load_context_data(filepath):
    if not os.path.exists(filepath):
        print(f"❌ [Error] File not found: {filepath}")
        print("   -> Run the Florence-2 context extractor script first!")
        sys.exit(1)

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        results = data.get("results", {})

        od_res = results.get("Object Detection", {})
        labels = od_res.get("labels", []) if isinstance(od_res, dict) else []
        unique_objs = list(set(labels))

        ocr_res = results.get("OCR", "")
        if isinstance(ocr_res, dict):
            texts = " ".join(ocr_res.get("labels", []))
        else:
            texts = str(ocr_res)

        return unique_objs, texts, data.get("source", "Unknown Image")

    except Exception as e:
        print(f"❌ [Error] Failed to parse JSON: {e}")
        sys.exit(1)


def encode_image_to_base64(image_path):
    if not os.path.exists(image_path):
        print(f"⚠️ [Image] Not found: {image_path} -- proceeding without image.")
        return None
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return encoded
    except Exception as e:
        print(f"⚠️ [Image] Failed to read/encode image: {e} -- proceeding without image.")
        return None


def query_llm(prompt, image_b64=None):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 2048,
            "num_predict": 150,
            "top_k": 20,
            "top_p": 0.0
        }
    }

    try:
        res = requests.post(f"{OLLAMA_API_URL}/api/generate", json=payload)
        return res.json().get("response", "<No response>")
    except Exception as e:
        return f"[Error] API Call failed: {e}"


def main():
    ensure_ollama_running()
    ensure_model_pulled()

    print(f"\n📂 [IO] Loading context from: {JSON_SOURCE}")
    objs, texts, img_source = load_context_data(JSON_SOURCE)

    obj_str = ", ".join(objs) if objs else "None detected"
    text_str = texts if texts.strip() else "None visible"

    print(f"   > Image (from Florence): {img_source}")
    print(f"   > Context: {len(obj_str)} objects, {len(text_str)} chars of text.")

    image_b64 = encode_image_to_base64(IMAGE_SOURCE)

    prompt_a = (
        f"Write a strictly factual description of this image. "
        f"Describe the objects, actions, and setting objectively. "
        f"Quote the text exactly as it appears on the signage/objects. "
        f"Do NOT start with 'Here is a caption' or 'This image shows'. "
        f"Start directly with the caption."
    )

    prompt_b = (
        f"Write a strictly factual description of this image. "
        f"Visually confirm and incorporate these specific details: "
        f"Objects present: [{objs}]. "
        f"Text visible: [{texts}]. "
        f"Instructions: Merge these details naturally into a single descriptive paragraph. "
        f"Quote the text exactly as it appears on the signage/objects. "
        f"Do NOT start with 'Here is a caption' or 'This image shows'. "
        f"Start directly with the caption."
    )

    print("\n" + "=" * 50)
    print("🧪 EXPERIMENT: BLIND VS. GROUNDED")
    print("=" * 50)

    print("\n--- 1. Blind Inference (No Context) ---")
    res_a = query_llm(prompt_a, image_b64=image_b64)
    print(f"\n{res_a.strip()}")

    print("\n" + "-" * 50)

    print("\n--- 2. Grounded Inference (With Explicit Context) ---")
    res_b = query_llm(prompt_b, image_b64=image_b64)
    print(f"\n{res_b.strip()}")

    print("\n" + "=" * 50)
    print("✅ Done.")

if __name__ == "__main__":
    main()