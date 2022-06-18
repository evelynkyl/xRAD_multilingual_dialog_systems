# Agents

This directory contains a variety of different agents which use ParlAI's interface. CORA is implemented by myself, the rest are from ParlAI. mT5 is implemented in `parlai_internal/agents/hugging_face`. 

There are many more agents in the [official repo](https://github.com/facebookresearch/ParlAI/tree/main/parlai/agentsI).


- **cora**: retrieval-augmented multilingual generative QA model
- **local_human**: receives human input from the terminal. used for interactive mode, e.g. `parlai/scripts/interactive.py`.

- **transformer**: both generative and retrieval-based transformer models
- **hugging_face**: includes GPT2, T5, and DialoGPT from HuggingFace. [`parlai_internal/agents/hugging_face`](https://github.com/evelynkyl/xRAD_multilingual_dialog_systems/main/parlai_internal/agents/hugging_face) contains mT5 as well.

