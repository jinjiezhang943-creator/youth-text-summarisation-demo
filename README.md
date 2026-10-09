# Youth Text Summarisation Pipeline with Phi-3

A privacy-safe demonstration of theme-level text summarisation using Microsoft Phi-3 Mini, PyTorch, and Hugging Face Transformers.

This repository is a sanitised standalone version of the summarisation module I developed as part of a university Industry Capstone project. It does not contain the original private dataset, other team members' code, or private repository history.

All responses included in this public repository are manually constructed examples and do not represent real participants.

---

## Project Objective

The project investigates how language models can be used to summarise large groups of short free-text survey responses.

In the full project pipeline, responses are first cleaned and assigned a **Main Category** and **Sub-category**. The summarisation module then groups responses by these labels and generates one concise theme-level summary for each group.

The main goal is to produce summaries that are:

- concise;
- faithful to the original responses;
- representative of the dominant ideas;
- able to preserve important less-frequent suggestions;
- suitable for structured downstream analysis.

### Pipeline

```text
Labelled Responses
        ↓
Main Category + Sub-category
        ↓
Grouped Responses
        ↓
Token Length Check
        ↓
Direct Summarisation
        or
Hierarchical Summarisation
        ↓
Theme-level Summary
        ↓
Human Evaluation
        ↓
Structured Output
```

---

## My Contribution

My responsibilities in the original team project included:

- contributing to manual labelling and thematic analysis of student responses;
- investigating text summarisation approaches including BART, FLAN-T5, and Phi-3;
- implementing and testing the Phi-3 summarisation module;
- designing System Prompts and User Prompts for concise and faithful theme-level summaries;
- processing responses grouped by Main Category and Sub-category;
- implementing token-aware direct and hierarchical summarisation for response groups of different sizes;
- designing a flexible sub-category mapping layer to support later manual consolidation of similar sub-categories;
- evaluating summaries using coverage, faithfulness, synthesis, clarity, and conciseness;
- documenting limitations, quality-control issues, and integration requirements.

---

## Model

The public demonstration uses:

```text
microsoft/Phi-3-mini-4k-instruct
```

The model is loaded through Hugging Face Transformers and runs locally using PyTorch.

Generation is deterministic because sampling is disabled:

```python
do_sample=False
```

The current prompt design instructs the model to:

- combine repeated ideas;
- distinguish dominant ideas from less frequent ideas;
- include important different suggestions;
- avoid listing responses one by one;
- avoid unsupported inferences;
- avoid adding information only because it appears in the category label;
- keep the final summary concise and neutral.

---

## Direct and Hierarchical Summarisation

Small response groups can be summarised directly if the complete prompt fits within the model's token budget.

Large groups may exceed the context window. Instead of truncating responses, the pipeline uses hierarchical summarisation.

### Direct Summarisation

```text
Grouped Responses
        ↓
Single Prompt
        ↓
Phi-3 Mini
        ↓
Final Summary
```

### Hierarchical Summarisation

```text
Large Response Group
        ↓
Token-aware Batching
        ↓
Batch Summary 1
Batch Summary 2
Batch Summary 3
...
        ↓
Intermediate Summaries
        ↓
Final Phi-3 Summarisation
        ↓
Final Theme-level Summary
```

This approach reduces the risk of losing responses through simple truncation and allows the summariser to process much larger response groups.

---

## Flexible Sub-category Mapping

The project supports later manual review and consolidation of similar sub-categories.

For example:

```python
SUBCATEGORY_MAPPING = {
    "Mental Well-being": "Mental Health and Wellbeing",
    "Mental Health Wellbeing": "Mental Health and Wellbeing",
    "Mental Health Improvement": "Mental Health and Wellbeing"
}
```

If the mapping is empty, the current sub-category labels are used directly.

This allows the summarisation pipeline to remain unchanged even if the upstream taxonomy is later revised.

---

## Project Structure

```text
youth-text-summarisation-demo/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── phi3_summarizer.py
├── sample_data/
│   └── example_responses.csv
└── examples/
    └── phi3_output.json
```

---

## Input Format

The input CSV contains the following columns:

| Column | Description |
|---|---|
| `response_id` | Synthetic response identifier |
| `grade` | Example school grade |
| `category` | Main response category |
| `subcategory` | More specific response theme |
| `comments` | Free-text response used for summarisation |

The included public sample contains manually constructed responses across several thematic groups.

The public sample is intentionally small and synthetic so that the repository remains privacy-safe and reproducible.

---

## Requirements

- Python 3.10 or later
- pandas
- PyTorch
- Hugging Face Transformers
- Accelerate

Install the required packages with:

```bash
python -m pip install -r requirements.txt
```

---

## Running the Demonstration

Run:

```bash
python src/phi3_summarizer.py
```

The program:

1. loads the synthetic response dataset;
2. loads Phi-3 Mini;
3. groups responses by category and subcategory;
4. constructs a summarisation prompt for each group;
5. generates one theme-level summary for each group;
6. displays the results in the terminal;
7. saves the structured results as JSON.

The generated results are saved to:

```text
examples/phi3_output.json
```

---

## Output Format

Each generated result contains:

```json
{
  "category": "Health",
  "subcategory": "Mental health support services",
  "num_responses": 5,
  "summary": "A model-generated theme-level summary.",
  "model": "microsoft/Phi-3-mini-4k-instruct"
}
```

---

## Evaluation Approach

The summarisation component is evaluated using manual review across five main criteria:

- **Coverage**  
  Whether the summary captures the major ideas in the source responses.

- **Faithfulness**  
  Whether the summary avoids adding unsupported information.

- **Synthesis**  
  Whether repeated responses are combined into higher-level ideas rather than rewritten individually.

- **Clarity**  
  Whether the summary is easy to understand.

- **Conciseness**  
  Whether the summary is brief and avoids unnecessary repetition.

Each criterion can be scored from 1 to 5.

The full project also pays attention to whether minority ideas are preserved and whether the relative importance of ideas is represented appropriately.

---

## Quality Control

Generated summaries are reviewed for common issues such as:

- unsupported details;
- over-generalisation;
- repeated ideas;
- excessive length;
- incomplete or truncated outputs;
- minority ideas being given too much or too little emphasis;
- poor summaries caused by mixed or overly broad upstream sub-categories.

One important finding from the project is that summarisation quality depends strongly on the quality of the upstream grouping stage.

A semantically mixed sub-category can produce a technically fluent summary that is still not useful for interpretation.

---

## Privacy

The original survey responses are not included because they may contain sensitive or personally identifiable information.

This repository contains:

- synthetic responses only;
- no student names;
- no original response identifiers;
- no private project files;
- no API keys or access tokens.

---

## Limitations

- The public repository uses synthetic data only and should not be interpreted as final performance on the private dataset.
- Summary quality depends on the quality of the Main Category and Sub-category labels produced by upstream stages.
- Semantically mixed or overly broad sub-categories may produce less focused summaries.
- Hierarchical summarisation reduces context-window limitations, but some information may still be over- or under-represented during summary-of-summaries generation.
- Human review remains important for checking faithfulness, coverage, and representativeness.
- Phi-3 Mini is used as a lightweight local model and may not match the performance of larger language models.

---

## Future Improvements

Possible future improvements include:

- improved sub-category consolidation after manual review;
- better preservation of minority ideas during hierarchical summarisation;
- frequency-aware weighting during summary-of-summaries generation;
- comparison with alternative local language models;
- automated quality checks for truncated or overly long summaries;
- integration into a complete end-to-end text analysis pipeline.

---

## Attribution

This repository represents only my summarisation contribution and a sanitised public demonstration.

It does not claim ownership of the complete team Capstone project or work completed by other team members.

