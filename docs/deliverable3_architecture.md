# Deliverable 3 Architecture

```mermaid
flowchart LR
    A["Processed prompt dataset<br/>results/collected_prompts.csv"] --> B["Train/test split"]
    B --> C["Model A baseline training<br/>DistilBERT checkpoint"]
    B --> D["Mutator stress generation<br/>wordnet / bert / t5 / roleplay / structural"]
    C --> E["Extended evaluation CSVs<br/>clean test + mutated_data"]
    D --> E
    E --> F["Deliverable 3 analysis<br/>threshold sweep + subgroup breakdowns"]
    F --> G["Balanced mode<br/>threshold 0.51"]
    F --> H["Strict mode<br/>threshold 0.85"]
    G --> I["Flask UI demo"]
    H --> I
    C --> I
    A --> J["Seeded jailbreak cache"]
    J --> I
```

## Notes

- The Deliverable 3 production recommendation is the calibrated `Model A` checkpoint.
- The adversarial/cache-backed branch remains in the repo as a research artifact and comparison point.
- Interface and report evidence:
  - [Refined UI](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/ui/app.py)
  - [Extended report PDF](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/reports/deliverable3_report.pdf)
  - [Deliverable 3 summary JSON](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/results/deliverable3/deliverable3_summary.json)
