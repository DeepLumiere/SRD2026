
import json
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.corpus import stopwords
import numpy as np
from math import pi

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
    nltk.download('averaged_perceptron_tagger')

STOPWORDS = set(stopwords.words('english'))
SMOOTH_FUNC = SmoothingFunction().method1

# Publication styling
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)  # Increased font scale for readability
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "figure.dpi": 300,
    "savefig.bbox": "tight",  # vital for preventing clipping
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "legend.frameon": True,
    "legend.fancybox": False
})


# =========================
# UTILITIES
# =========================

def safe_tokenize(text):
    if not isinstance(text, str): return []
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace("<pad>", " ").replace("</s>", "")
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def get_lexical_features(text):
    toks = safe_tokenize(text)
    if not toks: return 0, 0, 0.0
    tags = nltk.pos_tag(toks)
    n_count = len([w for w, t in tags if t.startswith('NN')])
    v_count = len([w for w, t in tags if t.startswith('VB')])
    complexity = len(set(toks)) / len(toks) if len(toks) > 0 else 0
    return n_count, v_count, complexity


# =========================
# METRICS LOGIC
# =========================

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


# =========================
# DATA LOADING
# =========================

def load_study_data(model_files, context_path, source_path):
    print("Loading Ground Truth Metadata...")
    with open(context_path, 'r', encoding='utf-8') as f:
        contexts = {item['image_id']: item for item in json.load(f)}

    print("Loading Reference Captions...")
    with open(source_path, 'r', encoding='utf-8') as f:
        sources = {}
        for item in json.load(f):
            raw_tokens = item.get('caption_tokens', [])
            sources[item['image_id']] = [t for t in raw_tokens if t not in ['<s>', '</s>']]

    results = []
    for model_name, filepath in model_files.items():
        print(f"Processing Model: {model_name}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    img_id = entry.get('image_id')
                    if img_id not in contexts or img_id not in sources: continue

                    settings = ['image_only', 'image_plus_context']
                    if model_name == "Florence-2": settings = ['caption']

                    for setting_key in settings:
                        raw_caption = entry.get(setting_key)
                        if not raw_caption: continue

                        if model_name == "Florence-2":
                            setting_label = "Baseline"
                        else:
                            setting_label = "Grounded" if "context" in setting_key else "Baseline"

                        cand_tokens = safe_tokenize(raw_caption)
                        n, v, comp = get_lexical_features(raw_caption)
                        ref_tokens = sources[img_id]

                        b1 = corpus_bleu([[ref_tokens]], [cand_tokens], weights=(1.0, 0, 0, 0),
                                         smoothing_function=SMOOTH_FUNC) * 100
                        met = meteor_score([ref_tokens], cand_tokens) * 100
                        iou = compute_iou(raw_caption, contexts[img_id])
                        ocr = compute_ocr_recall(raw_caption, contexts[img_id])

                        results.append({
                            "Model": model_name,
                            "Setting": setting_label,
                            "BLEU-1": b1, "METEOR": met,
                            "Grounding IoU": iou, "OCR Recall": ocr,
                            "Noun Count": n, "Verb Count": v, "Semantic Complexity": comp
                        })
        except FileNotFoundError:
            print(f"Warning: File {filepath} not found.")

    return pd.DataFrame(results)


# =========================
# PLOTTING FUNCTIONS
# =========================

def plot_bar_metric(df, metric, filename, title, y_label):
    plt.figure(figsize=(9, 7))  # Larger figure to prevent cut-off
    ax = sns.barplot(
        data=df, x="Model", y=metric, hue="Setting",
        palette="Paired", capsize=.1, errorbar=('ci', 95)
    )
    plt.title(title, fontweight='bold', pad=20)
    plt.ylabel(y_label, fontweight='bold')
    plt.xlabel("")
    plt.legend(title="Context Setting", loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Critical for preventing clipping
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()


def plot_3d_behavior_centroids(df):
    # """Plots only the centroid (mean) for each model/setting to reduce clutter."""
    # fig = plt.figure(figsize=(12, 10))  # Big square figure
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # Calculate Centroids
    # centroids = df.groupby(['Model', 'Setting'])[
    #     ['Noun Count', 'Verb Count', 'Semantic Complexity']].mean().reset_index()
    #
    # models = centroids['Model'].unique()
    # colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    # model_color_map = {m: c for m, c in zip(models, colors)}
    #
    # for _, row in centroids.iterrows():
    #     model = row['Model']
    #     setting = row['Setting']
    #     color = model_color_map[model]
    #
    #     # Distinct markers
    #     if "Grounded" in setting:
    #         marker = 'X'
    #         label_suffix = "(Grounded)"
    #         size = 200  # Make grounded markers significantly visible
    #     else:
    #         marker = 'o'
    #         label_suffix = "(Baseline)"
    #         size = 150
    #
    #     ax.scatter(
    #         row['Noun Count'], row['Verb Count'], row['Semantic Complexity'],
    #         label=f"{model} {label_suffix}",
    #         s=size, c=[color], marker=marker,
    #         edgecolors='k', linewidth=1.5, alpha=1.0
    #     )
    #
    # ax.set_xlabel('Avg Noun Density', labelpad=10)
    # ax.set_ylabel('Avg Verb Density', labelpad=10)
    # ax.set_zlabel('Avg Lexical Diversity', labelpad=10)
    # plt.title("Fig 4: Behavioral Shift (Centroids)", fontweight='bold', pad=20)
    #
    # # Move legend outside to prevent overlap with 3D box
    # plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), title="Model Configuration")
    #
    # plt.tight_layout()
    # plt.savefig("../graphics/fig4_behavioral_centroids.pdf", format='pdf', bbox_inches='tight')
    # plt.close()

    centroids = (
        df.groupby(["Model", "Setting"])[
            ["Noun Count", "Verb Count", "Semantic Complexity"]
        ]
        .mean()
        .reset_index()
    )

    # Create readable labels
    centroids["Label"] = centroids.apply(
        lambda r: f"{r['Model']} ({'Grounded' if 'Grounded' in r['Setting'] else 'Baseline'})",
        axis=1
    )

    # Sort for visual clarity (Baseline → Grounded per model)
    centroids = centroids.sort_values(
        by=["Model", "Setting"], ascending=[True, True]
    )

    # =======================
    # Plot
    # =======================
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(centroids))

    noun = centroids["Noun Count"]
    verb = centroids["Verb Count"]
    semantic = centroids["Semantic Complexity"]

    ax.barh(
        y_pos, noun,
        label="Noun Density",
        color="#4C72B0"
    )
    ax.barh(
        y_pos, verb,
        left=noun,
        label="Verb Density",
        color="#DD8452"
    )
    ax.barh(
        y_pos, semantic,
        left=noun + verb,
        label="Lexical Diversity",
        color="#55A868"
    )

    # =======================
    # Axes & Labels
    # =======================
    ax.set_yticks(y_pos)
    ax.set_yticklabels(centroids["Label"])
    ax.set_xlabel("Average Feature Magnitude")
    ax.set_title(
        "Fig 4: Behavioral Composition Shift (Centroid Analysis)",
        fontweight="bold",
        pad=12
    )

    # =======================
    # Legend
    # =======================
    ax.legend(
        loc="lower right",
        title="Linguistic Components"
    )

    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.savefig(
        "../graphics/fig4_behavioral_centroids.pdf",
        format="pdf"
    )
    plt.close()


def plot_balanced_radar(df, filename="../graphics/fig7_radar_balanced.pdf"):
    """Normalized Radar Chart with fixed layout clipping."""
    metrics = ["BLEU-1", "METEOR", "Grounding IoU", "OCR Recall"]

    # Group data
    data = df.groupby(["Model", "Setting"])[metrics].mean().reset_index()
    if data.empty: return

    # Normalize
    max_values = {m: data[m].max() * 1.15 for m in metrics}  # +15% buffer

    N = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Labels with padding to prevent clipping
    labels = [f"{m}\n(Max: {int(max_values[m])})" for m in metrics]
    plt.xticks(angles[:-1], labels, color='#333333', size=12, fontweight='bold')

    # Ensure labels aren't too close to the plot
    ax.tick_params(pad=20)

    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["25%", "50%", "75%", "100%"], color="grey", size=8)
    plt.ylim(0, 1.1)

    unique_models = data['Model'].unique()
    palette = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))
    model_colors = {m: c for m, c in zip(unique_models, palette)}

    for i, row in data.iterrows():
        model = row['Model']
        setting = row['Setting']
        color = model_colors[model]

        raw_vals = row[metrics].tolist()
        norm_vals = [v / max_values[m] for v, m in zip(raw_vals, metrics)]
        norm_vals += norm_vals[:1]

        if "Grounded" in setting:
            linestyle = '-'
            alpha = 0.1
            label = f"{model} (Grounded)"
            marker = 'o'
        else:
            linestyle = '--'
            alpha = 0.0
            label = f"{model} (Baseline)"
            marker = 'x'

        ax.plot(angles, norm_vals, linewidth=2.5, linestyle=linestyle,
                label=label, color=color, marker=marker, markersize=6)
        ax.fill(angles, norm_vals, color=color, alpha=alpha)

    plt.title("Fig 7: Normalized Model Capabilities", size=18, fontweight='bold', y=1.1)

    # Place legend clearly outside
    plt.legend(loc='upper right', bbox_to_anchor=(1.45, 1.05), title="Model & Setting", fontsize=10)

    # Extra margin for radar charts
    plt.subplots_adjust(top=0.85, right=0.75)
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()


def plot_heatmap(df):
    metrics = ["BLEU-1", "METEOR", "Grounding IoU", "OCR Recall"]
    pivot = df.groupby("Model")[metrics].mean()

    plt.figure(figsize=(9, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5)
    plt.title("Fig 6: Model Performance Overview", fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("../graphics/fig6_metrics_heatmap.pdf", format='pdf', bbox_inches='tight')
    plt.close()


# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":
    model_paths = {
        "Gemma3-12B": "../data/imgcapt/vision_ablation_checkpoint_gemma.jsonl",
        "Ministral3-8B": "../data/imgcapt/vision_ablation_checkpoint_ministral3.jsonl",
        "Qwen2.5-VL-7B": "../data/imgcapt/vision_ablation_checkpoint_qwen.jsonl",
        "Florence-2": "../data/imgcapt/vision_ablation_checkpoint_florence2.jsonl"
    }

    ctx_path = "../data/imgcapt/dataset_context_metadata.json"
    src_path = "../data/imgcapt/source_captions.json"

    df_results = load_study_data(model_paths, ctx_path, src_path)

    if not df_results.empty:
        print("\n--- Generating Clean Publication Figures (PDF) ---")

        plot_bar_metric(df_results, "BLEU-1", "../graphics/fig1_bleu_score.pdf",
                        "Fig 1: Linguistic Accuracy (BLEU-1)", "BLEU-1 Score (%)")

        plot_bar_metric(df_results, "METEOR", "../graphics/fig2_meteor_score.pdf",
                        "Fig 2: Semantic Consensus (METEOR)", "METEOR Score (%)")

        plot_bar_metric(df_results, "Grounding IoU", "../graphics/fig3_grounding_iou.pdf",
                        "Fig 3: Entity Grounding Precision", "Intersection over Union (%)")

        plot_bar_metric(df_results, "OCR Recall", "../graphics/fig5_ocr_recall.pdf",
                        "Fig 5: Text Recovery Performance", "Recall Rate (%)")

        plot_3d_behavior_centroids(df_results)
        plot_heatmap(df_results)
        plot_balanced_radar(df_results, "../graphics/fig7_radar_balanced.pdf")

        print("Done. All figures saved as PDF with clipped regions fixed.")
    else:
        print("No data processed.")
