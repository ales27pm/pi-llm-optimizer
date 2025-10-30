# Québecois Distillation Dataset

This folder contains a small seed dataset used for fine tuning the
student model.  Each line in `dataset_quebecois_distill.jsonl` is a
JSON object with at least a `user` field and additional metadata
describing the context and expected behaviour of the assistant.  After
labelling with the teacher model (`teacher_label.py`), the file
`labelled.jsonl` will have an extra `assistant` field containing the
teacher’s response.

## Fields

| Field           | Description                                                  |
|-----------------|--------------------------------------------------------------|
| `system_hint`   | Optional system prompt that sets context or style.         |
| `user`          | User’s message or question.                                |
| `requires_json` | If true, the assistant should respond in JSON.             |
| `tool_name`     | Name of the tool to call when `requires_json` is true.     |
| `tool_schema`   | JSON schema string defining the shape of the JSON output.  |
| `goal`          | Human‑readable description of the desired outcome.         |
| `tags`          | List of categories (conversation, math, tool, etc.).       |

## Usage

1. Review and edit `dataset_quebecois_distill.jsonl` to include your
   own prompts, contexts and tool calls.  Feel free to add new
   categories or alter the JSON schemas as needed.
2. Run `python desktop_distill/teacher_label.py --model <teacher> --input dataset/dataset_quebecois_distill.jsonl --output dataset/labelled.jsonl` to label the records.  The teacher will generate an
   `assistant` field for each entry.
3. Use `dataset/labelled.jsonl` as the input for the student training
   script.

The provided examples cover a variety of scenarios including
Québecois colloquialisms, mathematical reasoning, tool invocation
(emails, restaurants), honesty and curiosity.  In the latest update
we added tasks around slang expressions (parler de "boss des bécosses"),
explanations of idioms like "avoir les yeux dans la graisse de bines"
【459736271680283†L156-L166】【459736271680283†L170-L176】, cultural questions about the origin of *poutine*【916153969751195†L120-L133】, and new tool
calls (e.g. searching Hydro‑Québec invoices or local notes).  We also
introduced an additional file, `dataset_quebecois_conversation.jsonl`, which
focuses on **everyday conversation**.  These entries incorporate
contractions and vocabulary typical of spoken Québec French—such as
`tsé` for *tu sais*, `chu` for *je suis* and `pis` for *puis*【892647919746951†L260-L264】—and
common lexicon differences like **la fin de semaine** (weekend),
**magasiner** (to shop) and polite responses like "Bienvenue"
instead of "De rien"【892647919746951†L442-L471】.  We drew inspiration
from authentic dialogues where speakers use fillers like *ben* and
form yes/no questions with *t'es‑tu…?*【508168847965778†L96-L115】【508168847965778†L107-L110】.
These conversation prompts encourage the assistant to ask follow‑up
questions, clarify goals, and employ the everyday slang that makes
Québec French distinctive.  Feel free to expand the dataset with
many more examples tailored to your domain to achieve robust
performance.