# recall

recall is a backend-agnostic provenance system in Rust. It answers questions from an ingested corpus so that every claim cites the exact chunk versions it came from, is checked against them before anything is shown, and is stored with a full trace of how it was produced. When the sources do not support an answer, it declines instead of guessing.

The core is domain-free. Corpora, answering rules, golden sets and front ends belong to the application that uses it: a source connector, `explain.instructions` and its own config. Every model and store sits behind a port with a fake and contract tests, so a backend is an adapter, not a dependency of the pipeline.

- [Design](docs/design.md): the pipeline model, ports, provenance checks, trace and evaluation.
- [Conventions](AGENTS.md): the done checks and the coding rules.

## What it does

- **Ingest**: a `Source` port returns document snapshots; chunks are content-addressed (`ChunkRef {id, version}`), so a trace stays reproducible after the corpus changes. A markdown-directory connector ships; any other connector names itself and plugs in through `ingest::run_with` without a core or schema change.
- **Retrieve**: hybrid full-text (BM25 over prose, exact identifier tokens) and vector search in Postgres with pgvector, fused by reciprocal rank, with neighbouring sections as context and optional glossary expansion.
- **Decide and explain**: a judge (System One by default) classifies the question and answers yes/no questions with calibrated probabilities; an LLM writes the answer as claims whose citations are constrained to this run's chunks.
- **Provenance checks**: a vendor-neutral `Judge` port screens sources before they reach the LLM (injection, relevance, contradiction, evidence) and checks each claim against the chunks it cites (quoted spans, then supports / contradicts / says nothing). Failed claims decline the run or are stripped.
- **Audit**: every run, answered, declined or failed, is persisted with its trace: each step's model, prompt version, score and raw output, every admission screen and every claim verdict. `GET /v1/runs/{id}` returns it.
- **Evaluate**: `recall eval` scores a golden set per step (classification, recall@50, hit@5, precision, decline and failure rates, citation validity, faithfulness), gates against a baseline, and re-tunes thresholds from recorded scores.

- **Access control**: every run is asked on behalf of a principal; an `Authorizer` port compiles its roles into the chunks it may read, retrieval filters by it in SQL, and every step rechecks its sources, so a chunk the caller may not read never reaches a judge, the LLM, a citation or a decline. Decisions are in the trace.

Not built yet: Answer for `choice` and `ranking` (they take the `open` path).

## How a question is answered

```
Classify ∥ Retrieve → Rerank [→ Admit] [→ Answer] → Explain → Verify
```

```mermaid
sequenceDiagram
    actor C as Client
    participant S as recall serve
    participant P as Pipeline
    participant K as System One / Judge
    participant E as Embeddings
    participant DB as Postgres + pgvector
    participant L as Explain LLM

    C->>S: POST /v1/chat/completions (bearer key)
    S->>P: last user message as the question
    par Classify
        P->>K: one choice over boolean, choice, ranking, open
        K-->>P: category and its probability
    and Retrieve
        P->>E: embed the question
        E-->>P: query vector
        P->>DB: full text top 50 and vector top 50, fused by RRF
        DB-->>P: 50 chunks, with neighbouring sections as context
    end
    Note over P: Below classify.min_confidence, or choice and ranking: the open path
    opt rerank.judge names a judge (off by default)
        loop Rerank: each of the top 25 matches
            P->>K: does this text help answer the question?
            K-->>P: probability of yes
        end
    end
    Note over P: Rerank keeps the best 5 by retrieval rank,<br/>fused with the judge's when it judges
    opt admit.judge names a judge (off by default)
        loop Admit: each kept source
            P->>K: injection? relevant? contradicts the premise? usable evidence?
            K-->>P: four probabilities; excluded sources never reach Explain
        end
    end
    opt boolean
        P->>K: Answer: going by these sources, is the answer yes?
        K-->>P: verdict and its confidence, checked against the threshold
    end
    P->>L: Explain: question, 5 sources and any verdict, delimited as data
    L-->>P: claims as JSON citing chunk ids, a contradiction declines
    Note over P: Verify: an uncited claim, a citation outside this run's chunks,<br/>or a quoted span no cited chunk contains fails the claim
    opt verify.support is judged (off by default)
        loop each claim, its cited chunks in order
            P->>K: does this text support, contradict, or say nothing about the claim?
            K-->>P: the chosen relation and its probability
        end
    end
    Note over P: verify.publish: a failed claim declines the run,<br/>or with strip is left out
    P-->>S: run report (answered, declined or failed) and trace
    S->>DB: save the run and its trace
    S-->>C: answer with numbered citations and the run id
```

Each step is typed, so a pipeline with steps in the wrong order does not compile. Verify rejects any claim without a citation or citing a chunk the run did not retrieve and, when configured, any claim whose quoted spans or judged support its cited chunks do not back. Every run, including declines and failures, is stored with a trace of each step's model, prompt version, score, raw output, admission screens and claim verdicts: the audit record.

## Running locally

Requires Docker and Rust (the toolchain is pinned in `rust-toolchain.toml`; `rustup` installs it).

```sh
docker compose up -d --wait db     # Postgres + pgvector on localhost:5433
cp .env.example .env               # then fill it in, see Configuration
set -a; source .env; set +a

export RECALL_DOCS_DIR=/path/to/markdown
cargo run -- ingest
cargo run -- serve                 # API on RECALL_BIND
```

Ask a question:

```sh
curl -s localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-local-admin" \
  -H "Content-Type: application/json" \
  -d '{"model": "recall", "messages": [{"role": "user", "content": "Which table maps users to roles?"}]}'
```

The answer is in `choices[0].message.content`. The top-level `provenance` object carries the run id, outcome, and citations. Use `GET /v1/runs/{id}` for the full trace; the access policy decides whose runs a user may read.

### Running fully locally

`.env.example` is set up for local models; nothing calls AWS.

- **Embeddings and Explain: [Ollama](https://ollama.com).** Embeddings must have 1024 dimensions.

  ```sh
  brew install ollama && brew services start ollama
  ollama pull bge-m3
  ollama pull qwen3:14b
  ```

- **System One (Classify): [Kev](https://github.com/jaredpalmer/kev)** on Apple Silicon via MLX.

Switching embedding models requires `recall ingest --reembed`.

### Running on hosted models

Every model can run on [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/) with one API token: Explain on `@cf/qwen/qwen3.8-27b`, embeddings on `@cf/baai/bge-m3` (1024 dimensions), and System One on Clef Flash, `@cf/cloudflare/clef-flash`. Uncomment the "Hosted" block in `.env.example` (and comment out the local one), fill in the account id and token, then:

```sh
docker compose up -d --wait db-hosted   # its own database on port 5434; the hosted block's DATABASE_URL points at it
set -a; source .env; set +a        # in a fresh shell, so no variable from the old block lingers
cargo run -- models check          # one call to each model: its model id, endpoint, latency, and ok or the full error
cargo run -- ingest                # a fresh corpus embedded by the hosted model; `db` keeps the local one
cargo run -- eval run --golden your-golden.jsonl --out eval/results/workers-ai.json --concurrency 2
cargo run -- eval tune --results eval/results/workers-ai.json --candidates candidates.json
```

`recall models check` exits non-zero when any backend fails. Each line shows the endpoint called (scheme, host and path; no keys or query), so a URL left over from another block is visible. It names Bedrock, with its AWS region and credentials provider, whenever a port is on Bedrock, so a wrong `.env` is obvious before a long run.

Re-tune the thresholds: Clef Flash is calibrated differently from Kev, so the `classify.min_confidence` and `answer` thresholds in `config/recall.json` were tuned for another model's probabilities. `recall eval tune` re-applies candidate thresholds to the new run's recorded scores without calling any model; paste the best candidate into the config.

What the configuration relies on:

- **Chat and embeddings** use the `workers-ai` backend: `RECALL_*_ACCOUNT_ID`, `RECALL_*_KEY` (the API token) and the model id, with `Authorization: Bearer <token>`; a `RECALL_*_URL` left over from another block is rejected. Explain posts to Workers AI's [OpenAI-compatible](https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/) `.../accounts/{ACCOUNT_ID}/ai/v1/chat/completions`. Embeddings run on the native `.../accounts/{ACCOUNT_ID}/ai/run/@cf/baai/bge-m3` with `{"text": [...], "truncate_inputs": true}`, since the OpenAI-compatible route ignores `truncate_inputs`. A failure is reported by Workers AI's `errors`, with the model and route.
- **Long sections are embedded from their first 8,192 tokens**, bge-m3's input limit, as Ollama's bge-m3 does, so hosted and local vectors match. Without truncation such a section fails (`Sequence too long: 10670 > 8192`). The chunk keeps its id and full text, which BM25 and Explain still use. Each embed call logs one warning with the number of inputs estimated over the limit.
- **Embedding batches.** `@cf/baai/bge-m3` rejects a request over its 60,000-token [context](https://developers.cloudflare.com/workers-ai/models/bge-m3/) (`Max context reached ... but model supports only 60000`), and Workers AI counts a request as its text count times its longest text's tokens: every input is padded to the longest, so 35 sections with one of 2,916 tokens count as 102,060. The `workers-ai` embedder knows that limit and packs each request, still at most 64 texts, so that count times its longest text's estimate, at 3 bytes a token, which overcounts prose and markdown, stays under 50,000; the margin to 60,000 covers text that tokenizes denser. No setting changes it: `RECALL_EMBED_MAX_BATCH_TOKENS` is for `openai-compatible` servers and is rejected here, and an embedding model whose limit the adapter does not know is rejected at startup with the supported list (`@cf/baai/bge-m3`). A chunk estimated over the budget alone goes in a request of its own.
- **Reasoning effort.** [qwen3.8-27b](https://developers.cloudflare.com/workers-ai/models/qwen3.8-27b/) takes `reasoning_effort` `low`, `medium` or `xhigh` (the default). `RECALL_EXPLAIN_REASONING_EFFORT` sends it only when set. Use `medium`: `low` with a `json_schema` response format, which Explain always sends, [times out at 120 s](https://github.com/cloudflare/workers-sdk/issues/16165).
- **`max_tokens` includes the reasoning** on a reasoning model. Explain's budget is `explain.max_tokens` in the pipeline config: 2,048 in `config/recall.json`, enough for `qwen3:14b`'s answer alone, and 8,192 in `config/recall-workers-ai.json`, which the hosted block selects with `RECALL_CONFIG`. Use 8,192 for qwen3.8-27b at `reasoning_effort` `medium`: its reasoning comes out of the same budget before the answer, and at 2,048 it left too little, so the JSON answer was cut off mid-string (`EOF while parsing a string`). A reply stopped at the limit (`finish_reason length`, or Bedrock's stop reason `max_tokens`) fails the run with `answer cut off at max_tokens=N`, even when part of the answer arrived: a budget too small for the model is a configuration problem, not a question the sources cannot answer. Reasoning never reaches the answer: a separate `reasoning` field is not read, and a `<think>` block is dropped.
- **Schema shape.** The adapter sends OpenAI's `response_format.json_schema` form, `{name, strict, schema}`. Workers AI's [JSON Mode documentation](https://developers.cloudflare.com/workers-ai/features/json-mode/) shows the bare schema instead; if Workers AI rejects or ignores the OpenAI form, set `RECALL_EXPLAIN_SCHEMA_FORMAT=bare`.
- **Clef Flash** ([model page](https://developers.cloudflare.com/workers-ai/models/clef-flash/)) answers the System One request at `POST .../accounts/{ACCOUNT_ID}/ai/run/@cf/cloudflare/clef-flash`, with `model` set to `clef-flash`, inside Workers AI's `{result, success, errors, messages}` envelope. Its context is 24,576 tokens, and a longer text state is truncated to fit without an error. recall caps a state at 16,000 characters and logs a warning when a request's input tokens come within 512 of the limit. It costs $0.038 per million input tokens, and output is free.
- **system-one.dev** hosts Clef Flash too, behind the System One API (`POST https://system-one.dev/v1/systemone`, bearer key, 120 decision requests a minute by default, model `clef-flash` per `GET https://system-one.dev/v1/models`; [API docs](https://docs.system-one.dev/en/docs/api)). Use the commented alternative in `.env.example`.

## Commands

| Command | What it does |
|---|---|
| `recall serve` | OpenAI-compatible API: `/v1/models`, `/v1/chat/completions`, `/v1/runs/{id}` |
| `recall ingest` | Snapshot ingest from `RECALL_DOCS_DIR`; `--reembed` after changing the embedding model |
| `recall models check` | One call to the embedder, the Explain LLM and System One, with the same arguments as `serve` and `eval run`; prints each model, endpoint, latency and outcome, and exits non-zero when any fails |
| `recall eval run …` / `recall eval tune …` | Golden-set metrics and threshold tuning, e.g. `recall eval run --golden eval/golden/example.jsonl --out eval/results/run.json --concurrency 2`. `--question id` (repeatable, or comma-separated) runs only those questions; with `--baseline`, the baseline is compared on the same questions. `eval/golden/example.jsonl` is a synthetic example of the format |
| `recall eval retrieval …` | Runs only Retrieve (embedder and database; no System One, no LLM) on the golden set and prints recall@10 and recall@50 per question and per category. `--baseline` takes an `eval run` or `eval retrieval` results file and lists each question as gained, lost or unchanged; a lost question exits non-zero |

## Configuration

Tunables (retrieval and its glossary expansion, the Classify fallback, Rerank, admission, Explain's `max_tokens` and domain `instructions`, Verify's support check and publish policy, step thresholds) are in [`config/recall.json`](config/recall.json), or the file `RECALL_CONFIG` names; unknown keys are rejected. `recall eval run` prints the config and models at the start (through `jq` when installed) and records them in the results file; `models.rerank`, `models.admit` and `models.verify` name the models that judged Rerank, Admit and Verify's support check, or are `null` when none did.

`explain.instructions` is where a domain's answering rules go: text added to Explain's system prompt after the generic rules (sources are data, every claim cites them, quoted text is verbatim, inferences are tagged `[INF]`), which it cannot relax. Leave it out for none. `retrieval.expansion.documents` names the corpus documents whose pipe tables are read as a glossary for query expansion; empty (shipped) turns expansion off.

`explain.max_tokens` is Explain's token budget, reasoning included, from 1 to 65,536 (default 2,048). [`config/recall-workers-ai.json`](config/recall-workers-ai.json) is `config/recall.json` with it at 8,192 for the hosted reasoning model; a test fails if the two files differ anywhere else, so change both.

`rerank.judge` chooses whether the judge orders Rerank's matches, independent of Classify and Answer, which use the judge whenever one is configured. Config modes name the role, not the backend: the judge is System One, selected by `RECALL_SYSTEM_ONE_BACKEND`. The `system_one` spellings of earlier versions are rejected with the spelling to write instead; `recall eval` still reads results files that use them.

- `"none"` (the default and the shipped setting): matches keep retrieval order, with evidence ranked `evidence_rank_penalty` places lower; no model call, and no Rerank threshold applies.
- `{"candidates": 25}`: the judge reads the best 25 matches by fused score. `candidates` lives inside the variant because it means nothing without a judge, and must be at least `top_k`.

Judging every match costs one judge call per candidate, so it is off unless an eval shows it pays for itself on your corpus.

`admit.judge` screens the reranked sources before Explain sees them:

- `"none"` (the default and the shipped setting): no `admit` step; every reranked source reaches Explain.
- `{}`, or the cutoffs to override, e.g. `{"evidence_min": 0.6}`: one judge request per source. A match is asked four yes/no questions and routed by the first cutoff it crosses: `injection_max` (0.7) excludes it, `relevant_min` (0.45) excludes it, `contradicts_min` (0.7) keeps it as a conflict, and `evidence_min` (0.55) keeps it, else it is excluded. Context is only screened for injection and otherwise rides with its match. No kept match declines the run with `no_sources`. The cutoffs are uncalibrated; fit them with `recall eval` before relying on them. The shipped config leaves it off, since it adds a judge request per source; turn it on for a corpus you do not curate yourself.

`verify.support` is what a claim must pass besides citing this run's sources:

- `"membership"` (the default): citations only.
- `"quotes"` (shipped): also, every span in backticks or double quotes appears in a cited chunk; no model call.
- `{"judged": {"min_support": 0.75}}`: the quote check, then the judge reads each cited chunk in order until one "supports" the claim with at least `min_support`.

A check that names a judge never runs as a quieter one: with `RECALL_SYSTEM_ONE_BACKEND=none`, a config whose `rerank.judge`, `admit.judge` or `verify.support` needs a judge is a startup error listing each such key, and so is a `rerank` threshold while `rerank.judge` is `"none"` (the step is unscored). Without a judge every question is classified `open`. `recall serve` logs the assembled plan (`classify`, `rerank`, `admit`, `support` modes) at startup, `recall eval run` records it as `plan` in the results file, and each run's Classify, Rerank and Verify trace records carry the mode they ran in.

`verify.publish` is `"decline"` (the default and shipped: any failed claim declines the run) or `"strip"` (failed claims are left out and the rest published; the run declines only when none passes). Every verdict is in the Verify trace record, and `recall eval` reports the share of judged claims that are supported as `faithfulness`.

### Access control

[`config/access.json`](config/access.json), or the file `RECALL_ACCESS` names, is required by `serve`, `ingest` and `eval`. It defines roles (a sensitivity `clearance`, the `tenants` they read, and whose `runs`), the roles `everyone` holds, `members` per user id, and `labels` rules that relabel documents at ingestion by document-id prefix. A chunk is readable when its ACL is empty or names the user or one of its roles, and one of its roles clears its sensitivity and tenant; anything else, including a user the policy does not know, is denied. `recall eval run` and `recall eval retrieval` ask as `--as` (`RECALL_EVAL_AS`). The model is in [the design](docs/design.md#8-access-control).

See `.env.example`. Key variables: `DATABASE_URL`, `RECALL_API_KEYS`, `RECALL_ACCESS`, `RECALL_DOCS_DIR`, `RECALL_DOCS_BASE_URL`, `RECALL_DOCS_AUTHORITATIVE` / `RECALL_DOCS_EVIDENCE` / `RECALL_DOCS_SKIP` (which paths are authoritative, which are supporting evidence, and which are left out), and the model backends below.

Each model port names its backend, with no default. A backend's missing or unparsable setting (a URL is checked when parsed), or a setting it does not use, is a startup error naming the variable. Keys are never printed.

| Port | Backends | Variables |
|---|---|---|
| Embeddings | `bedrock`, `openai-compatible` (Ollama), `workers-ai` | `RECALL_EMBED_BACKEND`, `RECALL_EMBED_MODEL`; `openai-compatible`: `RECALL_EMBED_URL`, optional `RECALL_EMBED_KEY`, `RECALL_EMBED_MAX_BATCH_TOKENS` (a positive integer); `workers-ai`: `RECALL_EMBED_ACCOUNT_ID`, `RECALL_EMBED_KEY`, model `@cf/baai/bge-m3`, and no URL or batch budget |
| Explain | `bedrock`, `openai-compatible` (Ollama), `workers-ai` | `RECALL_EXPLAIN_BACKEND`, `RECALL_EXPLAIN_MODEL`; `openai-compatible`: `RECALL_EXPLAIN_URL`, optional `RECALL_EXPLAIN_KEY`; `workers-ai`: `RECALL_EXPLAIN_ACCOUNT_ID`, `RECALL_EXPLAIN_KEY`, a `@cf/...` model id, and no URL; both: optional `RECALL_EXPLAIN_REASONING_EFFORT` (`low`, `medium`, `xhigh`), `RECALL_EXPLAIN_SCHEMA_FORMAT` (`openai`, `bare`) |
| System One | `none`, `system-one` (Kev, system-one.dev), `workers-ai` (Clef) | `RECALL_SYSTEM_ONE_BACKEND`; `system-one`: `RECALL_SYSTEM_ONE_URL`, `RECALL_SYSTEM_ONE_MODEL`, optional `RECALL_SYSTEM_ONE_KEY`; `workers-ai`: `RECALL_SYSTEM_ONE_ACCOUNT_ID`, `RECALL_SYSTEM_ONE_MODEL`, `RECALL_SYSTEM_ONE_KEY`, and no URL: it always calls `api.cloudflare.com`, and a set `RECALL_SYSTEM_ONE_URL` (say, Kev's, left from the local block) is a startup error |

## Code layout

| Path | Contents |
|---|---|
| `src/domain.rs` | Shared types: questions, chunks and their versions, citations, provenance, probabilities |
| `src/ingest/` | The `Source` port, its contract suite and fakes, versioning and embedding; `markdown.rs` is the markdown-directory connector |
| `src/retrieval/` | The `Retriever` port and the hybrid Postgres retriever |
| `src/llm/` | Model ports (`Embedder`, `ChatModel`, `SystemOne`, `Judge`), their adapters and fakes |
| `src/pipeline/` | Typed pipeline, steps (`Classify`, `Retrieve`, `Rerank`, `Admit`, `Answer`, `Explain`, `Verify`), trace and report |
| `src/store/` | The `RunStore` port and Postgres stores |
| `src/server/` | OpenAI-compatible HTTP API |
| `src/eval/` | Golden-set metrics, baseline gate, threshold tuning |
| `migrations/` | Postgres schema |

## Development

With the database running:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo nextest run        # cargo install cargo-nextest --locked
cargo test --doc         # nextest does not run doctests; the compile_fail ones guard the pipeline types
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
cargo audit
```

Conventions are in [AGENTS.md](AGENTS.md).
