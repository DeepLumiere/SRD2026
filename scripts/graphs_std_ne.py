import json
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.corpus import stopwords
import numpy as np

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
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "legend.frameon": True,
    "legend.fancybox": False
})

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

def plot_bar_metric(df, metric, filename, title, y_label):
    """Standard single-metric bar plot."""
    plt.figure(figsize=(9, 7))
    sns.barplot(
        data=df, x="Model", y=metric, hue="Setting",
        palette="Paired", capsize=.1, errorbar=('ci', 95)
    )
    plt.title(title, fontweight='bold', pad=20)
    plt.ylabel(y_label, fontweight='bold')
    plt.xlabel("")
    plt.legend(title="Context Setting", loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()

def plot_linguistic_delta(df, filename="../graphics/fig4_behavioral_delta.pdf"):
    """
    Fig 4 Replacement: Density Delta Plot.
    Calculates the shift in *proportion* of Nouns/Verbs,
    neutralizing the effect of shorter captions.
    """
    # 1. Create Density Metrics (Normalize by sum of N+V to approximate content length)
    # We use (Noun + Verb) as a proxy for 'content length' since we don't have total tokens here.
    df = df.copy()
    df['Content Length'] = df['Noun Count'] + df['Verb Count']

    # Avoid division by zero
    df = df[df['Content Length'] > 0]

    df['Noun Density'] = df['Noun Count'] / df['Content Length']
    df['Verb Density'] = df['Verb Count'] / df['Content Length']
    # Complexity is already a ratio (TTR), so we keep it as is.

    target_features = {
        "Noun Density": "Noun Focus",
        "Verb Density": "Action Focus",
        "Semantic Complexity": "Lexical Diversity"
    }

    # 2. Aggregate Mean values per Model/Setting
    # We aggregate the DENSITIES now, not the counts.
    df_agg = df.groupby(["Model", "Setting"])[list(target_features.keys())].mean().reset_index()

    # 3. Pivot for Baseline vs Grounded comparison
    df_pivot = df_agg.pivot(index="Model", columns="Setting", values=target_features.keys())

    # Flatten columns
    df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]

    plot_data = []

    # 4. Calculate Relative Change (%)
    for model in df_pivot.index:
        # Check if model has both settings
        check_col = f"{list(target_features.keys())[0]}_Grounded"
        if check_col not in df_pivot.columns or pd.isna(df_pivot.loc[model, check_col]):
            continue

        for feat, label in target_features.items():
            b_val = df_pivot.loc[model, f"{feat}_Baseline"]
            g_val = df_pivot.loc[model, f"{feat}_Grounded"]

            if b_val == 0:
                pct_change = 0
            else:
                pct_change = ((g_val - b_val) / b_val) * 100

            plot_data.append({
                "Model": model,
                "Feature": label,
                "Change (%)": pct_change
            })

    df_final = pd.DataFrame(plot_data)

    if df_final.empty:
        print("Skipping Fig 4: Data insufficient for delta calculation.")
        return

    # 5. Plotting
    plt.figure(figsize=(10, 6))
    plt.axhline(0, color='black', linewidth=1, alpha=0.5)

    ax = sns.barplot(
        data=df_final,
        x="Model",
        y="Change (%)",
        hue="Feature",
        palette=['#4C72B0', '#DD8452', '#55A868'],
        edgecolor="black"
    )

    plt.title("Fig 4: Behavioral Shift (Change in Compositional Density)", fontweight='bold', pad=15)
    plt.ylabel("Change in Density (%)", fontweight='bold')
    plt.xlabel("")

    # Add values
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=16, fontweight='bold')

    plt.legend(title="Behavioral Component", loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()

# def plot_multi_metric_summary(df, filename="../graphics/fig7_multimetric_summary.pdf"):
#     """
#     Figure 7 Replacement:
#     Grouped Bar Plot comparing all major metrics side-by-side.
#     """
#     metrics = ["BLEU-1", "METEOR", "Grounding IoU", "OCR Recall"]
#
#     # Convert to Long Format for Seaborn
#     df_long = df.melt(
#         id_vars=["Model", "Setting"],
#         value_vars=metrics,
#         var_name="Metric",
#         value_name="Score"
#     )
#
#     # Initialize the figure
#     plt.figure(figsize=(12, 7))
#
#     # Create the grouped bar plot
#     # x = Metric, y = Score, hue = Model
#     # We facet by Setting (Baseline vs Grounded) to keep it clean,
#     # OR we include Setting in the Hue.
#     # Let's create a combined hue "Model (Setting)" to show everything on one plot
#     # but that might be crowded.
#     # Best approach: FacetGrid or just Hue=Model and average the settings
#     # IF the user wants a general overview.
#     # BUT, the distinction is usually important.
#     # Let's do: X=Metric, Hue=Model, Style=Setting (using pattern) -- patterns are hard in seaborn.
#     # Let's do: X=Metric, Hue=Model+Setting.
#
#     df_long["Configuration"] = df_long["Model"] + " (" + df_long["Setting"] + ")"
#
#     # Filter/Sort to keep colors consistent
#     df_long = df_long.sort_values(by=["Model", "Setting"])
#
#     ax = sns.barplot(
#         data=df_long,
#         x="Metric",
#         y="Score",
#         hue="Configuration",
#         palette="tab10",
#         errorbar=None  # Remove error bars for cleaner summary view
#     )
#
#     plt.title("Fig 7: Comprehensive Model Performance", fontweight='bold', pad=15)
#     plt.ylabel("Score (0-100)", fontweight='bold')
#     plt.xlabel("")
#     plt.ylim(0, 105)
#
#     # Legend placement
#     plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Configuration")
#
#     # Add values on top of bars
#     for i in ax.containers:
#         ax.bar_label(i, fmt='%.0f', padding=3, fontsize=9)
#
#     plt.grid(axis='y', linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     plt.savefig(filename, format='pdf', bbox_inches='tight')
#     plt.close()
def plot_multi_metric_summary(df, filename="../graphics/fig7_multimetric_summary.pdf"):
    """
    Figure 7 Replacement:
    2x2 Subplot Grid comparing all major metrics.
    """
    metrics = ["BLEU-1", "METEOR", "Grounding IoU", "OCR Recall"]
    titles = {
        "BLEU-1": "Linguistic Accuracy (BLEU-1)",
        "METEOR": "Semantic Consensus (METEOR)",
        "Grounding IoU": "Spatial Grounding (IoU)",
        "OCR Recall": "Text Recovery (OCR Recall)"
    }

    # Initialize 2x2 Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Consistent Color Palette
    palette = "Paired"  # Light/Dark pairs are good for Baseline/Grounded

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Plot individual metric
        sns.barplot(
            data=df,
            x="Model",
            y=metric,
            hue="Setting",
            palette=palette,
            ax=ax,
            errorbar=('ci', 95),  # Keep confidence intervals if available
            capsize=0.1
        )

        # Styling per subplot
        ax.set_title(titles[metric], fontweight='bold', fontsize=16)
        ax.set_ylabel("Score (%)" if i % 2 == 0 else "")  # Only show Y-label on left plots
        ax.set_xlabel("")
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # Remove individual legends (we will add a global one later)
        if ax.get_legend():
            ax.get_legend().remove()

        # Add values on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=2, fontsize=16)

    # Create a Single Global Legend at the top
    # We grab handles/labels from the first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        title="Experimental Setting",
        fontsize=16,
        title_fontsize=16,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()

def plot_heatmap(df):
    metrics = ["BLEU-1", "METEOR", "Grounding IoU", "OCR Recall"]
    pivot = df.groupby("Model")[metrics].mean()

    plt.figure(figsize=(10, 10))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5)
    plt.title("Fig 6: Model Performance Overview", fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("../graphics/fig6_metrics_heatmap.pdf", format='pdf', bbox_inches='tight')
    plt.close()


# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":
    # Ensure directories exist
    import os

    if not os.path.exists("../graphics"):
        os.makedirs("../graphics")

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

        # Basic Single Metrics
        plot_bar_metric(df_results, "BLEU-1", "../graphics/fig1_bleu_score.pdf",
                        "Fig 1: Linguistic Accuracy (BLEU-1)", "BLEU-1 Score (%)")

        plot_bar_metric(df_results, "METEOR", "../graphics/fig2_meteor_score.pdf",
                        "Fig 2: Semantic Consensus (METEOR)", "METEOR Score (%)")

        plot_bar_metric(df_results, "Grounding IoU", "../graphics/fig3_grounding_iou.pdf",
                        "Fig 3: Entity Grounding Precision", "Intersection over Union (%)")

        plot_bar_metric(df_results, "OCR Recall", "../graphics/fig5_ocr_recall.pdf",
                        "Fig 5: Text Recovery Performance", "Recall Rate (%)")

        plot_linguistic_delta(df_results, "../graphics/fig4_behavioral_delta.pdf")
        # Heatmap
        # plot_heatmap(df_results)

        # UPDATED FIGURE 7: Grouped Bar Plot (Replaced Radar)
        plot_multi_metric_summary(df_results, "../graphics/fig7_multimetric_summary.pdf")

        print("Done. All figures saved.")
    else:
        print("No data processed.")