import json
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import numpy as np
from collections import defaultdict

# Metric Imports
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.corpus import stopwords

# Check for ROUGE/CIDEr libraries
try:
    from rouge_score import rouge_scorer

    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("Warning: 'rouge_score' not found. ROUGE-L will be skipped.")


# =========================
# CONFIGURATION & SETUP
# =========================

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading necessary NLTK datasets...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

STOPWORDS = set(stopwords.words('english'))
SMOOTH_FUNC = SmoothingFunction().method1

# Publication styling
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 300,
    "legend.frameon": True,
    "legend.fancybox": False
})


# =========================
# UTILITIES & METRICS
# =========================

def safe_tokenize(text):
    if not isinstance(text, str): return []
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace("<pad>", " ").replace("</s>", "")
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def compute_iou(generated_text, context_json):
    gen_tokens = set(safe_tokenize(generated_text))
    gt_tokens = set()
    for obj in context_json.get('objects', []):
        gt_tokens.update(safe_tokenize(obj.get('class', '')))
    for txt in context_json.get('texts', []):
        gt_tokens.update(safe_tokenize(txt))
    if not gt_tokens: return 0.0
    intersection = len(gen_tokens & gt_tokens)
    union = len(gen_tokens | gt_tokens)
    return (intersection / union * 100) if union > 0 else 0.0


def compute_ocr_recall(generated_text, context_json):
    gen_tokens = set(safe_tokenize(generated_text))
    gt_text_tokens = set()
    for txt in context_json.get('texts', []):
        gt_text_tokens.update(safe_tokenize(txt))
    if not gt_text_tokens: return 0.0
    matches = gen_tokens & gt_text_tokens
    return (len(matches) / len(gt_text_tokens) * 100)

def compute_rouge_l(candidate, references):
    if not HAS_ROUGE: return 0.0
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    # ROUGE against multiple refs: usually take the max F-measure
    scores = [scorer.score(ref, candidate)['rougeL'].fmeasure for ref in references]
    return max(scores) * 100 if scores else 0.0


# =========================
# DATA LOADING
# =========================

def load_study_data(model_files, context_path, source_path):
    print("Loading Ground Truth Metadata...")
    with open(context_path, 'r', encoding='utf-8') as f:
        contexts = {item['image_id']: item for item in json.load(f)}

    print("Loading Reference Captions...")
    with open(source_path, 'r', encoding='utf-8') as f:
        sources_tokens = {}  # For BLEU/METEOR (tokens)
        sources_raw = defaultdict(list)  # For CIDEr/ROUGE (strings)

        for item in json.load(f):
            img_id = item['image_id']
            # Raw string construction for ROUGE/CIDEr
            raw_text = " ".join(item.get('caption_tokens', [])).replace("<s>", "").replace("</s>", "").strip()
            sources_raw[img_id].append(raw_text)

            # Token list for BLEU
            tokens = [t for t in item.get('caption_tokens', []) if t not in ['<s>', '</s>']]
            sources_tokens[img_id] = tokens

    all_data = []

    for model_name, filepath in model_files.items():
        print(f"Processing Model: {model_name}...")

        # We need to collect all hypotheses first for Batch CIDEr
        model_hyps = {'w/o Context': [], 'w/ Context': []}

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    img_id = entry.get('image_id')
                    if img_id not in contexts or img_id not in sources_tokens: continue

                    settings = ['image_only', 'image_plus_context']
                    if model_name == "Florence-2": settings = ['caption']

                    for setting_key in settings:
                        raw_caption = entry.get(setting_key)
                        if not raw_caption: continue

                        setting_label = "w/o Context" if model_name == "Florence-2" or "context" not in setting_key else "w/ Context"

                        # Store for batch processing
                        model_hyps[setting_label].append({
                            'image_id': img_id,
                            'caption': raw_caption,
                            'context': contexts[img_id],
                            'refs_tokens': sources_tokens[img_id],
                            'refs_raw': sources_raw[img_id]
                        })

            # Process collected hypotheses
            for setting_label, items in model_hyps.items():
                if not items: continue

                for item in items:
                    img_id = item['image_id']
                    cand_text = item['caption']
                    cand_tokens = safe_tokenize(cand_text)
                    ref_tokens = item['refs_tokens']
                    ref_raw_list = item['refs_raw']

                    # 1. BLEU-1
                    b1 = corpus_bleu([[ref_tokens]], [cand_tokens], weights=(1.0, 0, 0, 0),
                                     smoothing_function=SMOOTH_FUNC) * 100

                    # 2. METEOR
                    met = meteor_score([ref_tokens], cand_tokens) * 100

                    # 3. ROUGE-L
                    rouge = compute_rouge_l(cand_text, ref_raw_list)

                    # 5. Grounding
                    iou = compute_iou(cand_text, item['context'])
                    ocr = compute_ocr_recall(cand_text, item['context'])

                    all_data.append({
                        "Model": model_name,
                        "Setting": setting_label,
                        "BLEU-1": b1,
                        "METEOR": met,
                        "ROUGE-L": rouge,
                        "Content Coverage": iou,
                        "OCR Recall": ocr
                    })

        except FileNotFoundError:
            print(f"Warning: File {filepath} not found.")

    return pd.DataFrame(all_data)


# =========================
# PLOTTING
# =========================

def plot_metric_bars(df, filename="../graphics/fig4_metric_bars.pdf"):
    """
    Generates bar charts in a 2-2-1 layout (3 rows) with the last plot centered.
    Includes a global legend at the top.
    """
    metrics = ["BLEU-1", "METEOR", "ROUGE-L", "Content Coverage", "OCR Recall"]
    valid_metrics = [m for m in metrics if m in df.columns and df[m].sum() > 0]

    # Prepare data
    df_melt = df.melt(id_vars=["Model", "Setting"],
                      value_vars=valid_metrics,
                      var_name="Metric",
                      value_name="Score")

    # Initialize Figure
    fig = plt.figure(figsize=(12, 12))

    # GridSpec: 3 rows, 4 columns
    # We use 4 columns to easily center the bottom plot (spanning cols 1-3)
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

    # Define positions for 5 plots
    axes_locs = [
        gs[0, 0:2], gs[0, 2:4],  # Row 1: Left, Right
        gs[1, 0:2], gs[1, 2:4],  # Row 2: Left, Right
        gs[2, 1:3]  # Row 3: Center (spans middle 2 cols)
    ]

    palette = {'w/o Context': '#b8b8b8', 'w/ Context': '#1a80bb'}

    global_handles = None
    global_labels = None

    for i, metric in enumerate(valid_metrics):
        if i >= len(axes_locs): break

        # Create subplot
        ax = fig.add_subplot(axes_locs[i])

        # Plot
        sns.barplot(
            data=df_melt[df_melt["Metric"] == metric],
            x="Model",
            y="Score",
            hue="Setting",
            palette=palette,
            edgecolor="black",
            errorbar=None,
            ax=ax
        )

        # 1. Capture legend handles/labels from the FIRST plot only
        if i == 0:
            global_handles, global_labels = ax.get_legend_handles_labels()

        # 2. Remove the individual plot legend (Seaborn adds it by default)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        # Titles and Labels
        ax.set_title(metric, fontweight='bold', size=14)
        ax.set_xlabel("")

        # Y-label logic: Left side plots (0, 2) and the centered bottom plot (4) need labels
        if i in [0, 2, 4]:
            ax.set_ylabel("Score")
        else:
            ax.set_ylabel("")

    # --- ADD GLOBAL LEGEND ---
    # We use the handles captured from the first plot to ensure colors match exactly
    if global_handles and global_labels:
        fig.legend(global_handles, global_labels,
                   loc='upper center',
                   bbox_to_anchor=(0.5, 0.93),  # Positioned just below title
                   ncol=2,
                   frameon=False,
                   fontsize=14)

    # Adjust layout to make room for legend
    plt.subplots_adjust(top=0.88)

    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Metric bars saved to {filename}")

if __name__ == "__main__":
    model_paths = {
        "Gemma3-12B": "../data/imgcapt/vision_ablation_checkpoint_gemma.jsonl",
        "Ministral3-8B": "../data/imgcapt/vision_ablation_checkpoint_ministral3.jsonl",
        "Qwen2.5-VL-7B": "../data/imgcapt/vision_ablation_checkpoint_qwen.jsonl",
        "Florence-2": "../data/imgcapt/vision_ablation_checkpoint_florence2.jsonl"
    }

    ctx_path = "../data/imgcapt/dataset_context_metadata.json"
    src_path = "../data/imgcapt/source_captions.json"

    # # Load and Compute
    # df_results = load_study_data(model_paths, ctx_path, src_path)
    #
    # if not df_results.empty:
    #     # Plot Bar Metrics
    #     plot_metric_bars(df_results)
    #
    #     # Save Aggregate Table
    #     agged = df_results.groupby(["Model", "Setting"]).mean()
    #     print("\nAggregate Results:")
    #     print(agged)
    #     agged.to_csv("../data/imgcapt/aggregate_results.csv")
    # else:
    #     print("No data processed.")
    csv_path = "../data/imgcapt/aggregate_results.csv"

    try:
        print(f"Loading pre-computed data from {csv_path}...")

        # Read the CSV.
        # Since 'to_csv' usually saves the index (Model, Setting), read_csv will pick them up.
        df_results = pd.read_csv(csv_path)

        # Plot directly using the loaded data
        if not df_results.empty:
            plot_metric_bars(df_results)
            print("Plot regeneration complete.")
        else:
            print("Dataframe is empty.")

    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}.")
        print("Please run the calculation step once to generate the CSV file.")