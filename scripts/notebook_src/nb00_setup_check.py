"""Source for notebooks/00_setup_check.ipynb."""

from _cells import SETUP, code, md

FILENAME = "00_setup_check.ipynb"

CELLS = [
md("""
# 00 · Setup check

**Northfield Health chart assistant** (synthetic data only) · **Agenda slot:** before 9:00 / end of the RAG talk · **Time box:** 10 min

**Goals.** Confirm, before Lab 1 starts, that:
- your `.env` loads and `WORKSHOP_USER` is set (every LangSmith resource you create is scoped by it)
- the chat model answers and the embeddings model returns vectors (hosted or local Ollama)
- your LangSmith key works, and **a trace actually lands** in your own project

Prefer a terminal? `uv run python scripts/check_setup.py` runs the same checks with a PASS/FAIL line each.

> Everything in this lab is **synthetic / fictional**: no real patients, PHI, medications, or clinical guidance.
"""),
md("## 1. Load the environment\n\nThis cell **fails loudly** if `WORKSHOP_USER` is unset. Fix `.env`, restart the kernel, and re-run."),
code(SETUP),
md("## 2. Chat model call"),
code('''from utils.models import get_model

reply = get_model().invoke("Reply with the single word: ready")
print(LAB["model"], "→", reply.content)'''),
md("## 3. Embeddings call"),
code('''from utils.models import get_embeddings

vector = get_embeddings().embed_query("most recent creatinine")
print(LAB["embeddings"], "→ dimension", len(vector))'''),
md("## 4. LangSmith key and a trace that lands\n\nWe trace one tiny function, then read it back by ID. If this passes, open the printed link: that's **your** project."),
code('''import time
import uuid

from langsmith import Client, traceable

client = Client()
run_id = uuid.uuid4()


@traceable(name="setup_check")
def ping(x: str) -> str:
    return f"pong {x}"


ping("chart-qa", langsmith_extra={"run_id": run_id, "project_name": LAB["langsmith_project"]})
client.flush()
for _ in range(20):
    try:
        run = client.read_run(run_id)
        break
    except Exception:
        time.sleep(1.5)
else:
    raise TimeoutError("Trace did not land. Check LANGSMITH_API_KEY / LANGSMITH_WORKSPACE_ID.")
print("✅ trace landed in", LAB["langsmith_project"])
print("🔗", client.get_run_url(run=run))'''),
md("## 5. The same checks as one command"),
code('''import subprocess
import sys

print(subprocess.run([sys.executable, str(_root / "scripts" / "check_setup.py")], capture_output=True, text=True).stdout)'''),
md("""
## What you just learned

- Every notebook starts with `init_lab()`: it loads `.env`, enforces `WORKSHOP_USER`, and sends traces to **your** project (`chart-qa-<you>`).
- Models are switched in one place: `LABS_MODEL`, `LABS_EMBEDDINGS`, and `LABS_JUDGE_MODEL` (see `utils/models.py`). Hosted and local Ollama both work.
- A trace is the unit of observability you'll use all day. Click the link above and look around.

**Next:** `01_build_rag.ipynb`.
"""),
]
