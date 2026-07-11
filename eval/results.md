# Evaluation Results

- **Hit Rate:** 82%
- **Answer Relevance (1–5):** 5.0
- **Faithfulness (1–5):** 5.0
- **LLM calls:** 29  ·  **Duration:** 107.8s

| # | Question | Multi-turn | Hit | Relevance | Faithfulness | Notes |
|---|----------|------------|-----|-----------|--------------|-------|
| richmond-thresholds | What are the competitive bidding thresholds at the University of Richmond? |  | ✅ | 5 | 5 |  |
| richmond-capital-equipment | What qualifies as capital equipment in the Richmond procurement policy and what is its minimum purchase price? |  | ✅ | 5 | 5 |  |
| richmond-card-invoice | Can a University of Richmond purchase card be used to pay an invoice? |  | ✅ | 5 | 5 |  |
| richmond-tech-purchases | Which department manages technology purchases such as hardware and software at the University of Richmond? |  | ✅ | 5 | 5 |  |
| oracle-order-vs-requisition | What is the difference between an order and a requisition in Oracle Procurement? |  | ✅ | 5 | 5 |  |
| oracle-po-types | What purchase order types does Oracle Purchasing provide? |  | ✅ | 5 | 5 |  |
| oracle-requisition-lifecycle | What does the requisition life cycle refer to in Oracle Procurement? |  | ✅ | 5 | 5 |  |
| oracle-reassign-requisition | Can I reassign a requisition that was created by someone else in Oracle Procurement? |  | ✅ | 5 | 5 |  |
| multiturn-oracle-requisition | What statuses can it have during approval? | yes | ❌ | 5 | 5 |  |
| multiturn-richmond-thresholds | What is required for purchases above the highest threshold? | yes | ✅ | 5 | 5 |  |
| cross-doc-approval-limit | What is the approval limit for purchases? |  | ❌ | 5 | 5 |  |
| out-of-scope-refusal | What is the capital of France? |  | — | 5 | 5 | refused correctly |
