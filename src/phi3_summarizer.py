from pathlib import Path
import json
import math

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "sample_data" / "example_responses.csv"
OUTPUT_FILE = PROJECT_ROOT / "examples" / "phi3_output.json"

MAX_INPUT_TOKENS = 2800
MAX_NEW_TOKENS = 180

# This public demo uses the existing sub-category labels.
# In the full project, this mapping can be updated after
# manual review and consolidation of similar sub-categories.
SUBCATEGORY_MAPPING = {}


# ============================================================
# Load Data
# ============================================================

def load_data(path):
    """
    Load the synthetic public demonstration dataset.
    """

    df = pd.read_csv(path)

    required_columns = {
        "response_id",
        "grade",
        "category",
        "subcategory",
        "comments",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df = df.dropna(subset=["comments"]).copy()

    df["comments"] = (
        df["comments"]
        .astype(str)
        .str.strip()
    )

    df = df[df["comments"] != ""].copy()

    # Preserve the original label for traceability.
    df["original_subcategory"] = df["subcategory"]

    # Allows later consolidation without changing
    # the summarisation logic.
    df["final_subcategory"] = (
        df["subcategory"]
        .replace(SUBCATEGORY_MAPPING)
    )

    return df


# ============================================================
# Load Model
# ============================================================

def load_model():
    """
    Load Phi-3 Mini using the native Hugging Face Transformers implementation.
    """

    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager",
    )

    model.eval()

    print("Model loaded successfully.")

    return tokenizer, model


# ============================================================
# Prompt Construction
# ============================================================

def build_prompt(
    tokenizer,
    main_category,
    sub_category,
    responses,
    stage="direct",
):
    """
    Build a chat prompt for either direct summarisation
    or final hierarchical synthesis.
    """

    response_text = "\n".join(
        f"- {response}"
        for response in responses
    )

    system_message = (
        "You summarise groups of short student wellbeing survey responses. "
        "Write a concise theme-level summary in 1-2 sentences. "
        "Focus only on ideas explicitly supported by the responses. "
        "Avoid broad conclusions, unsupported interpretations, "
        "or information that appears only in the category labels."
    )

    if stage == "final":
        task_instruction = """
The text below contains intermediate summaries generated from
different batches of responses.

Produce one final theme-level summary.

Requirements:
- Combine repeated ideas across the batch summaries.
- Give the most emphasis to ideas that appear across multiple batches.
- Use wording such as "some responses" for less frequent ideas.
- Do not give equal emphasis to ideas that appear only rarely.
- Do not repeat the same idea in different sentences.
- Do not add new information.
- Keep the final answer to no more than 2 sentences.
"""
    else:
        task_instruction = """
Summarise the student responses below.

Requirements:
- Combine repeated ideas.
- Include important different suggestions.
- Distinguish dominant ideas from less frequent ideas.
- Only describe an idea as a main theme if it appears repeatedly.
- Use wording such as "some responses" for less frequent ideas.
- Do not give equal emphasis to ideas that appear only rarely.
- Do not repeat the same idea in different sentences.
- Do not list responses one by one.
- Do not add information that is not explicitly supported.
- Do not infer missing details, causes, purposes, services, or outcomes.
- If a detail is not stated in the responses, leave it out.
- Keep the final answer to no more than 2 sentences.
"""

    messages = [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": f"""
Main category: {main_category}
Sub-category: {sub_category}

{task_instruction}

Text to summarise:
{response_text}

Summary:
""",
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ============================================================
# Token Utilities
# ============================================================

def count_tokens(tokenizer, text):
    """
    Count tokens without truncating the input.
    """

    return len(
        tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )["input_ids"]
    )


def split_responses_into_batches(
    tokenizer,
    main_category,
    sub_category,
    responses,
):
    """
    Split a large response group into token-aware batches.

    Each batch is kept below MAX_INPUT_TOKENS.
    """

    batches = []
    current_batch = []

    for response in responses:

        candidate_batch = current_batch + [response]

        prompt = build_prompt(
            tokenizer,
            main_category,
            sub_category,
            candidate_batch,
            stage="direct",
        )

        token_count = count_tokens(
            tokenizer,
            prompt,
        )

        if (
            token_count > MAX_INPUT_TOKENS
            and current_batch
        ):
            batches.append(current_batch)
            current_batch = [response]

        else:
            current_batch = candidate_batch

    if current_batch:
        batches.append(current_batch)

    return batches


# ============================================================
# Generation
# ============================================================

def generate_summary(
    tokenizer,
    model,
    prompt,
):
    """
    Generate one summary from a prepared prompt.
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False,
    )

    device = next(model.parameters()).device

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    input_length = inputs["input_ids"].shape[1]

    if input_length > MAX_INPUT_TOKENS:
        raise ValueError(
            f"Prompt contains {input_length} tokens, "
            f"which exceeds the configured limit of "
            f"{MAX_INPUT_TOKENS}."
        )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][input_length:]

    summary = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return summary.strip()


# ============================================================
# Direct Summarisation
# ============================================================

def direct_summarise(
    tokenizer,
    model,
    main_category,
    sub_category,
    responses,
):
    """
    Directly summarise a response group that fits
    within the token budget.
    """

    prompt = build_prompt(
        tokenizer,
        main_category,
        sub_category,
        responses,
        stage="direct",
    )

    summary = generate_summary(
        tokenizer,
        model,
        prompt,
    )

    return summary


# ============================================================
# Hierarchical Summarisation
# ============================================================

def hierarchical_summarise(
    tokenizer,
    model,
    main_category,
    sub_category,
    responses,
):
    """
    Summarise large groups using token-aware batching,
    followed by a final summary of intermediate summaries.
    """

    batches = split_responses_into_batches(
        tokenizer,
        main_category,
        sub_category,
        responses,
    )

    print(
        f"  Hierarchical summarisation: "
        f"{len(batches)} batches"
    )

    intermediate_summaries = []

    for batch_number, batch in enumerate(
        batches,
        start=1,
    ):

        print(
            f"    Processing batch "
            f"{batch_number}/{len(batches)} "
            f"({len(batch)} responses)"
        )

        batch_summary = direct_summarise(
            tokenizer,
            model,
            main_category,
            sub_category,
            batch,
        )

        intermediate_summaries.append(
            batch_summary
        )

    final_prompt = build_prompt(
        tokenizer,
        main_category,
        sub_category,
        intermediate_summaries,
        stage="final",
    )

    final_summary = generate_summary(
        tokenizer,
        model,
        final_prompt,
    )

    return final_summary, len(batches)


# ============================================================
# Automatic Strategy Selection
# ============================================================

def summarise_group(
    tokenizer,
    model,
    main_category,
    sub_category,
    responses,
):
    """
    Automatically choose direct or hierarchical
    summarisation based on token length.
    """

    direct_prompt = build_prompt(
        tokenizer,
        main_category,
        sub_category,
        responses,
        stage="direct",
    )

    token_count = count_tokens(
        tokenizer,
        direct_prompt,
    )

    print(
        f"  Responses: {len(responses)} | "
        f"Prompt tokens: {token_count}"
    )

    if token_count <= MAX_INPUT_TOKENS:

        summary = direct_summarise(
            tokenizer,
            model,
            main_category,
            sub_category,
            responses,
        )

        return {
            "summary": summary,
            "summary_method": "direct",
            "batch_count": 1,
            "input_tokens": token_count,
        }

    summary, batch_count = hierarchical_summarise(
        tokenizer,
        model,
        main_category,
        sub_category,
        responses,
    )

    return {
        "summary": summary,
        "summary_method": "hierarchical",
        "batch_count": batch_count,
        "input_tokens": token_count,
    }


# ============================================================
# Process All Groups
# ============================================================

def process_groups(
    df,
    tokenizer,
    model,
):
    """
    Generate one summary for each
    Main Category + Sub-category group.
    """

    results = []

    grouped = df.groupby(
        [
            "category",
            "final_subcategory",
        ],
        sort=True,
    )

    total_groups = grouped.ngroups

    print(
        f"\nNumber of thematic groups: "
        f"{total_groups}\n"
    )

    for group_number, (
        (main_category, sub_category),
        group,
    ) in enumerate(
        grouped,
        start=1,
    ):

        responses = (
            group["comments"]
            .dropna()
            .astype(str)
            .tolist()
        )

        print(
            f"[{group_number}/{total_groups}] "
            f"{main_category} -> "
            f"{sub_category}"
        )

        result = summarise_group(
            tokenizer,
            model,
            main_category,
            sub_category,
            responses,
        )

        results.append(
            {
                "category": main_category,
                "subcategory": sub_category,
                "num_responses": len(responses),
                "summary_method": (
                    result["summary_method"]
                ),
                "batch_count": (
                    result["batch_count"]
                ),
                "summary": result["summary"],
                "model": MODEL_NAME,
            }
        )

        print(
            f"  Method: "
            f"{result['summary_method']}"
        )

        print(
            f"  Summary: "
            f"{result['summary']}\n"
        )

    return results


# ============================================================
# Save Results
# ============================================================

def save_results(
    results,
    output_path,
):
    """
    Save structured summaries as JSON.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nResults saved to: "
        f"{output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Youth Text Summarisation Demo"
    )

    print(
        "=" * 40
    )

    df = load_data(
        INPUT_FILE
    )

    print(
        f"Loaded {len(df)} synthetic responses."
    )

    tokenizer, model = load_model()

    results = process_groups(
        df,
        tokenizer,
        model,
    )

    save_results(
        results,
        OUTPUT_FILE,
    )

    print(
        "\nSummarisation complete."
    )


if __name__ == "__main__":
    main()