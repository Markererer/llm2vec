"""
Generate confusion matrix for DeBERTa frozen EWT POS tagging model.
This script loads a checkpoint and generates confusion matrices for analysis.

Usage:
    python3 generate_confusion_matrix_ewt.py /path/to/checkpoint-3000 /path/to/output_dir
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)
from sklearn.metrics import confusion_matrix, classification_report


def load_checkpoint_and_config(checkpoint_path):
    """Load model and configuration from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load config to get label mappings
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    id2label = config.get("id2label", {})
    label2id = config.get("label2id", {})
    
    # Convert string keys to int for id2label
    id2label = {int(k): v for k, v in id2label.items()}
    
    print(f"Loaded {len(id2label)} labels")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Load model
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    return model, tokenizer, id2label, label2id, device


def load_test_dataset():
    """Load Universal Dependencies EWT test set."""
    print("Loading Universal Dependencies EWT test dataset...")
    raw_datasets = load_dataset(
        "universal_dependencies",
        "en_ewt",
        split="test"
    )
    print(f"Loaded {len(raw_datasets)} test examples")
    return raw_datasets


def get_predictions(model, tokenizer, dataset, id2label, label2id, device, max_seq_length=512, use_upos=True):
    """Get model predictions on the test set.
    
    Args:
        use_upos: If True, use UPOS tags (17 coarse-grained categories).
                  If False, use XPOS tags (50 fine-grained categories).
    """
    all_predictions = []
    all_labels = []
    all_upos_labels = []  # Store UPOS labels separately
    
    print(f"Running inference on {len(dataset)} examples...")
    print(f"Using {'UPOS (universal)' if use_upos else 'XPOS (fine-grained)'} tags")
    
    # Use hardcoded UPOS ID-to-name mapping
    upos_id_to_name = get_upos_id_to_name_mapping()
    print(f"Using standard UD UPOS mapping with {len(upos_id_to_name)} labels")
    
    for idx, example in enumerate(dataset):
        if idx % 100 == 0:
            print(f"Processing example {idx}/{len(dataset)}")
        
        tokens = example["tokens"]
        xpos_tags = example["xpos"]  # Fine-grained tags (what model was trained on)
        upos_tags_raw = example["upos"]  # Coarse-grained universal tags (IDs)
        
        # Convert UPOS integer IDs to string names
        upos_tags = [upos_id_to_name.get(uid, "X") for uid in upos_tags_raw]
        
        # Tokenize
        tokenized = tokenizer(
            tokens,
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Align predictions with original labels
        word_ids = tokenized.word_ids(batch_index=0)
        predictions = predictions[0].cpu().numpy()
        
        previous_word_idx = None
        for word_idx, pred_id in zip(word_ids, predictions):
            if word_idx is not None and word_idx != previous_word_idx:
                # Get the true XPOS label (what model predicts)
                true_xpos = xpos_tags[word_idx]
                true_upos = upos_tags[word_idx]
                
                if true_xpos in label2id:
                    all_predictions.append(pred_id)
                    all_labels.append(label2id[true_xpos])
                    # Store UPOS (should be string now after conversion above)
                    all_upos_labels.append(true_upos)
                previous_word_idx = word_idx
    
    print(f"Collected {len(all_predictions)} token predictions")
    
    if use_upos:
        # Map XPOS predictions AND true labels to UPOS using the SAME mapping
        print("Mapping fine-grained XPOS predictions to coarse-grained UPOS...")
        xpos_to_upos = create_xpos_to_upos_mapping()
        
        upos_predictions = []
        upos_true = []
        unknown_xpos_pred = set()
        unknown_xpos_true = set()
        
        for pred_id, true_id, true_upos_dataset in zip(all_predictions, all_labels, all_upos_labels):
            # Convert predicted XPOS to UPOS
            pred_xpos = id2label[pred_id]
            pred_upos = xpos_to_upos.get(pred_xpos, None)
            
            if pred_upos is None:
                unknown_xpos_pred.add(pred_xpos)
                pred_upos = "X"
            
            # Convert true XPOS to UPOS using the SAME mapping (not dataset UPOS)
            true_xpos = id2label[true_id]
            true_upos = xpos_to_upos.get(true_xpos, None)
            
            if true_upos is None:
                unknown_xpos_true.add(true_xpos)
                true_upos = "X"
            
            upos_predictions.append(pred_upos)
            upos_true.append(true_upos)
        
        if unknown_xpos_pred:
            print(f"WARNING: Found {len(unknown_xpos_pred)} unknown XPOS tags in predictions: {unknown_xpos_pred}")
        if unknown_xpos_true:
            print(f"WARNING: Found {len(unknown_xpos_true)} unknown XPOS tags in true labels: {unknown_xpos_true}")
        
        print(f"Sample mappings - First 10 predictions:")
        for i in range(min(10, len(all_predictions))):
            pred_xpos = id2label[all_predictions[i]]
            true_xpos = id2label[all_labels[i]]
            print(f"  Pred: {pred_xpos}->{upos_predictions[i]} | True: {true_xpos}->{upos_true[i]} | Match: {upos_predictions[i]==upos_true[i]}")
        
        return upos_predictions, upos_true
    else:
        return np.array(all_predictions), np.array(all_labels)


def get_upos_id_to_name_mapping():
    """Return the standard Universal Dependencies UPOS integer-to-string mapping.
    Based on UD v2 specification with 17 universal POS tags."""
    return {
        0: "ADJ",      # adjective
        1: "ADP",      # adposition
        2: "ADV",      # adverb
        3: "AUX",      # auxiliary
        4: "CCONJ",    # coordinating conjunction
        5: "DET",      # determiner
        6: "INTJ",     # interjection
        7: "NOUN",     # noun
        8: "NUM",      # numeral
        9: "PART",     # particle
        10: "PRON",    # pronoun
        11: "PROPN",   # proper noun
        12: "PUNCT",   # punctuation
        13: "SCONJ",   # subordinating conjunction
        14: "SYM",     # symbol
        15: "VERB",    # verb
        16: "X"        # other
    }


def create_xpos_to_upos_mapping():
    """Create mapping from XPOS (fine-grained) to UPOS (universal) tags.
    Based on Penn Treebank to Universal Dependencies mapping."""
    return {
        # Nouns
        "NN": "NOUN", "NNS": "NOUN", "NNP": "PROPN", "NNPS": "PROPN",
        
        # Verbs
        "VB": "VERB", "VBD": "VERB", "VBG": "VERB", "VBN": "VERB", 
        "VBP": "VERB", "VBZ": "VERB",
        
        # Adjectives
        "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ",
        
        # Adverbs
        "RB": "ADV", "RBR": "ADV", "RBS": "ADV", "WRB": "ADV",
        
        # Pronouns
        "PRP": "PRON", "PRP$": "PRON", "WP": "PRON", "WP$": "PRON",
        
        # Determiners
        "DT": "DET", "PDT": "DET", "WDT": "DET",
        
        # Prepositions/Adpositions
        "IN": "ADP", "TO": "ADP",
        
        # Conjunctions
        "CC": "CCONJ",
        
        # Numbers
        "CD": "NUM",
        
        # Particles
        "RP": "ADP", "POS": "PART",
        
        # Auxiliary
        "MD": "AUX",
        
        # Existential there
        "EX": "PRON",
        
        # Interjections
        "UH": "INTJ",
        
        # Other special tags
        "ADD": "X",  # Email/web address
        "GW": "X",   # Goes with (multi-word expression)
        
        # Symbols
        "SYM": "SYM", "#": "SYM", "$": "SYM",
        
        # Punctuation - all map to PUNCT
        ".": "PUNCT", ",": "PUNCT", ":": "PUNCT", "``": "PUNCT", "''": "PUNCT",
        "-LRB-": "PUNCT", "-RRB-": "PUNCT", "HYPH": "PUNCT",
        "NFP": "PUNCT", "AFX": "PUNCT", "XX": "X",
        
        # Other punctuation
        "!": "PUNCT", "\"": "PUNCT", "%": "PUNCT", "&": "PUNCT", "'": "PUNCT",
        "(": "PUNCT", ")": "PUNCT", "*": "PUNCT", "+": "PUNCT", "-": "PUNCT",
        "/": "PUNCT", ";": "PUNCT", "<": "PUNCT", "=": "PUNCT", ">": "PUNCT",
        "?": "PUNCT", "@": "PUNCT", "^": "PUNCT", "`": "PUNCT", "{": "PUNCT",
        "|": "PUNCT", "}": "PUNCT", "~": "PUNCT",
        
        # Foreign words / Other
        "FW": "X", "LS": "X",
    }


def plot_full_confusion_matrix(y_true, y_pred, labels, output_path, normalize=False, is_upos=False):
    """Plot full confusion matrix."""
    # For UPOS, we need to work with string labels
    if is_upos:
        # Correct UD UPOS ordering (17 classes)
        upos_order = [
            "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
            "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
            "SCONJ", "SYM", "VERB", "X"
        ]

        # Filter out labels not present in predictions (should be all 17)
        labels_present = [l for l in upos_order if l in set(y_true) or l in set(y_pred)]

        cm = confusion_matrix(y_true, y_pred, labels=labels_present)
        label_names = labels_present

    else:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
        label_names = [labels[i] for i in range(len(labels))]
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix (UPOS)' if is_upos else 'Normalized Confusion Matrix'
        filename = 'confusion_matrix_upos_normalized.png' if is_upos else 'confusion_matrix_normalized.png'
    else:
        fmt = 'd'
        title = 'Confusion Matrix - UPOS Tags (Counts)' if is_upos else 'Confusion Matrix (Counts)'
        filename = 'confusion_matrix_upos_counts.png' if is_upos else 'confusion_matrix_counts.png'
    
    # Create figure - smaller for UPOS (17 classes vs 50)
    figsize = (12, 10) if is_upos else (20, 18)
    fontsize = 10 if is_upos else 8
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=is_upos,  # Annotate UPOS (readable), don't annotate XPOS (too many)
        fmt=fmt,
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        square=True
    )
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)
    plt.tight_layout()
    
    output_file = os.path.join(output_path, filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file}")
    plt.close()


def plot_top_confusions(y_true, y_pred, labels, output_path, top_n=15, is_upos=False):
    """Plot confusion matrix for top-N most confused classes."""
    # For UPOS, we need to work with string labels
    if is_upos:
        # Use correct UPOS ordering
        upos_order = [
            "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
            "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
            "SCONJ", "SYM", "VERB", "X"
        ]
        unique_labels = [l for l in upos_order if l in set(y_true) or l in set(y_pred)]
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        label_names = unique_labels
        
        # Find classes with most errors (highest error rate)
        error_info = []
        for i, label in enumerate(unique_labels):
            total = cm[i, :].sum()
            if total > 0:
                correct = cm[i, i]
                error_count = total - correct
                error_rate = error_count / total
                error_info.append((i, label, error_rate, error_count))
            else:
                error_info.append((i, label, 0, 0))
    else:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
        
        # Find classes with most errors
        error_info = []
        for i in range(len(labels)):
            total = cm[i, :].sum()
            if total > 0:
                correct = cm[i, i]
                error_count = total - correct
                error_rate = error_count / total
                error_info.append((i, labels[i], error_rate, error_count))
            else:
                error_info.append((i, labels[i], 0, 0))
        label_names = [labels[i] for i in range(len(labels))]
    
    # Sort by error count (absolute), then by error rate as tiebreaker
    error_info.sort(key=lambda x: (x[3], x[2]), reverse=True)
    
    # Get top-N classes
    top_classes = [x[0] for x in error_info[:top_n]]
    top_labels = [x[1] for x in error_info[:top_n]]
    
    # Extract submatrix
    cm_subset = cm[np.ix_(top_classes, top_classes)]
    
    # Normalize
    cm_subset = cm_subset.astype('float') / cm_subset.sum(axis=1)[:, np.newaxis]
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_subset,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=top_labels,
        yticklabels=top_labels,
        cbar_kws={'label': 'Proportion'},
        square=True
    )
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    title = f'Top {top_n} Most Confused UPOS Classes' if is_upos else f'Top {top_n} Most Confused Classes'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    suffix = 'upos' if is_upos else 'xpos'
    output_file = os.path.join(output_path, f'confusion_matrix_{suffix}_top{top_n}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file}")
    plt.close()


def save_classification_report(y_true, y_pred, labels, output_path, is_upos=False):
    """Save detailed classification report."""
    if is_upos:
        # For UPOS, use correct ordering
        upos_order = [
            "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
            "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
            "SCONJ", "SYM", "VERB", "X"
        ]
        unique_labels = [l for l in upos_order if l in set(y_true) or l in set(y_pred)]
        report = classification_report(
            y_true,
            y_pred,
            labels=unique_labels,
            digits=4
        )
        filename = 'classification_report_upos.txt'
    else:
        label_names = [labels[i] for i in range(len(labels))]
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=label_names,
            digits=4
        )
        filename = 'classification_report_xpos.txt'
    
    output_file = os.path.join(output_path, filename)
    with open(output_file, 'w') as f:
        f.write(report)
    print(f"Saved {output_file}")
    
    # Also print to console
    print("\nClassification Report:")
    print(report)


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 generate_confusion_matrix_ewt.py /path/to/checkpoint /path/to/output_dir")
        print("Example: python3 generate_confusion_matrix_ewt.py /home/maass/scripts/output/word-task/pos_tags/deberta_ewt_frozen/checkpoint-3000 /home/maass/scripts/analysis/confusion_matrices")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    print(f"Output will be saved to: {output_path}")
    
    # Load model and data
    model, tokenizer, id2label, label2id, device = load_checkpoint_and_config(checkpoint_path)
    test_dataset = load_test_dataset()
    
    # Get predictions - UPOS (universal, 17 classes)
    print("\n" + "="*60)
    print("GENERATING UPOS (UNIVERSAL) CONFUSION MATRICES")
    print("="*60)
    y_pred_upos, y_true_upos = get_predictions(
        model, tokenizer, test_dataset, id2label, label2id, device, use_upos=True
    )
    
    # Calculate accuracy for UPOS
    correct = sum(1 for pred, true in zip(y_pred_upos, y_true_upos) if pred == true)
    accuracy_upos = correct / len(y_pred_upos)
    print(f"\nUPOS Test Accuracy: {accuracy_upos:.4f} ({accuracy_upos*100:.2f}%)")
    
    # Generate UPOS visualizations
    print("\nGenerating UPOS confusion matrices...")
    
    # 1. Full UPOS confusion matrix (counts)
    plot_full_confusion_matrix(y_true_upos, y_pred_upos, id2label, output_path, 
                               normalize=False, is_upos=True)
    
    # 2. Full UPOS confusion matrix (normalized)
    plot_full_confusion_matrix(y_true_upos, y_pred_upos, id2label, output_path, 
                               normalize=True, is_upos=True)
    
    # 3. Top-10 most confused UPOS classes
    plot_top_confusions(y_true_upos, y_pred_upos, id2label, output_path, 
                       top_n=10, is_upos=True)
    
    # 4. Save UPOS classification report
    save_classification_report(y_true_upos, y_pred_upos, id2label, output_path, is_upos=True)
    
    # Optional: Also generate XPOS (fine-grained) matrices
    print("\n" + "="*60)
    print("GENERATING XPOS (FINE-GRAINED) CONFUSION MATRICES")
    print("="*60)
    y_pred_xpos, y_true_xpos = get_predictions(
        model, tokenizer, test_dataset, id2label, label2id, device, use_upos=False
    )
    
    accuracy_xpos = (y_pred_xpos == y_true_xpos).mean()
    print(f"\nXPOS Test Accuracy: {accuracy_xpos:.4f} ({accuracy_xpos*100:.2f}%)")
    
    print("\nGenerating XPOS confusion matrices...")
    
    # 5. Top-15 most confused XPOS classes
    plot_top_confusions(y_true_xpos, y_pred_xpos, id2label, output_path, 
                       top_n=15, is_upos=False)
    
    # 6. Top-10 most confused XPOS classes
    plot_top_confusions(y_true_xpos, y_pred_xpos, id2label, output_path, 
                       top_n=10, is_upos=False)
    
    # 7. Save XPOS classification report
    save_classification_report(y_true_xpos, y_pred_xpos, id2label, output_path, is_upos=False)
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nOutput files saved to: {output_path}")
    print(f"\nUPOS files (17 classes - recommended for thesis):")
    print(f"  - confusion_matrix_upos_counts.png")
    print(f"  - confusion_matrix_upos_normalized.png")
    print(f"  - confusion_matrix_upos_top10.png")
    print(f"  - classification_report_upos.txt")
    print(f"\nXPOS files (50 classes - for detailed analysis):")
    print(f"  - confusion_matrix_xpos_top10.png")
    print(f"  - confusion_matrix_xpos_top15.png")
    print(f"  - classification_report_xpos.txt")


if __name__ == "__main__":
    main()
