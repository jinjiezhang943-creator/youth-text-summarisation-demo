# Phi-3 Youth Response Summariser

A privacy-safe demonstration of generative text summarisation using Microsoft Phi-3 Mini and Hugging Face Transformers.

This repository is a sanitised standalone version of the summarisation module I developed as part of a university team Capstone project. It does not contain the original private dataset, other team members' code or the private repository history.

All responses included in this public repository are manually constructed examples and do not represent real participants.

## Project Objective

The project investigates how a language model can summarise groups of short free-text survey responses.

The summariser receives responses that have already been cleaned and grouped into categories and subcategories. It then generates a concise theme-level summary for each group.

```text
Cleaned Responses
        ↓
Category and Subcategory Grouping
        ↓
Prompt Construction
        ↓
Phi-3 Mini Inference
        ↓
Structured JSON Output
```

## My Contribution

My responsibilities in the original team project included:

* investigating suitable text summarisation methods;
* implementing and testing the Phi-3 summarisation module;
* designing prompts for concise theme-level summaries;
* processing responses grouped by category and subcategory;
* preparing an initial manual evaluation dataset;
* evaluating summaries for coverage, faithfulness, synthesis, clarity and conciseness;
* documenting current limitations and integration requirements.

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

The prompt instructs the model to:

* combine repeated ideas;
* include different suggestions;
* produce a one- or two-sentence summary;
* avoid listing responses individually;
* avoid introducing unsupported information.

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

## Input Format

The input CSV contains the following columns:

| Column        | Description                               |
| ------------- | ----------------------------------------- |
| `response_id` | Synthetic response identifier             |
| `grade`       | Example school grade                      |
| `category`    | Main response category                    |
| `subcategory` | More specific response theme              |
| `comments`    | Free-text response used for summarisation |

The included file contains 25 synthetic responses across five thematic groups.

## Requirements

* Python 3.10 or later
* pandas
* PyTorch
* Hugging Face Transformers
* Accelerate

Install the required packages with:

```bash
python -m pip install -r requirements.txt
```

## Running the Demonstration

Run:

```bash
python src/phi3_summarizer.py
```

The program:

1. loads the synthetic response dataset;
2. loads the Phi-3 Mini model;
3. groups responses by category and subcategory;
4. generates one summary for each thematic group;
5. displays the results in the terminal;
6. saves the structured results as JSON.

The generated results are saved to:

```text
examples/phi3_output.json
```

## Output Format

Each generated result contains:

```json
{
  "category": "Community support",
  "subcategory": "Access to essentials",
  "num_responses": 5,
  "summary": "A model-generated theme-level summary.",
  "model": "microsoft/Phi-3-mini-4k-instruct"
}
```

## Evaluation Approach

The original experiment used manual review across five criteria:

* coverage;
* faithfulness;
* synthesis;
* clarity;
* conciseness.

Each criterion was scored from 1 to 5. This was an initial functional evaluation using small manually constructed response groups and should not be interpreted as final performance on the complete private dataset.

## Privacy

The original survey responses are not included because they may contain sensitive or personally identifiable information.

This repository contains:

* synthetic responses only;
* no student names;
* no original response identifiers;
* no private project files;
* no API keys or access tokens.

## Limitations

The current implementation has been tested on small grouped-response examples. Large groups may exceed the model context window and require batching or hierarchical summarisation.

The quality of generated summaries also depends on the quality of the earlier data-cleaning and theme-grouping stages.

## Attribution

This repository represents only my summarisation contribution and a sanitised public demonstration. It does not claim ownership of the complete team Capstone project or work completed by other team members.

