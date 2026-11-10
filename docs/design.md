# Drew AI (Drew as a Service) — v1 Design

Drew AI answers questions about SAP and ARM internals using Drew Steinfatt's pre-audited markdown documentation as the knowledge source. A small, fast "System One" model answers the **what** (yes/no, a choice, or a ranking); an LLM answers the **why**. Every answer carries provenance. v1 is a developer-facing Slack bot plus an OpenAI-compatible HTTP API.

Status: design agreed; crate foundation in place (shared types, pipeline machinery, schema, module boundaries with fakes). Confidence thresholds live in `config/drew.json` with starting values; the evaluation harness tunes them.

## 1. v1 scope

| In v1 | Deferred |
|---|---|
| Developer-facing Slack bot (Socket Mode) and OpenAI-compatible HTTP API | Customer access, Gaia UI |
| Batch ingestion of a Drew-audited markdown directory | Event-driven ingestion, revocation SLAs |
| Demo for internal developers (Drew reviews corpus and declines) | Broader rollout before launch gate |
| Permission/sensitivity labels **recorded** on every chunk | RBAC/RLS **enforcement** |
| Dockerfile + env config | EKS / Aurora / GitOps deployment |

Guiding constraints:

- **Provenance is mandatory.** No claim, option, or ranked item exists without a source.
- **System One never invents.** Options and ranked items come only from the user prompt or retrieved chunks.
- **Everything is auditable.** Every run, including declines and failures, is persisted with a full trace.
- **Invalid pipelines do not compile.** Each category's pipeline contains only the steps it needs.

## 2. Corpus and ingestion

Sources:

- A directory of **pre-audited markdown files**. Each file is one document; each ATX heading section is one chunk. Optional `DREW_DOCS_BASE_URL` turns paths into citation links.
- Two layers, chosen by path under the root: `DREW_DOCS_AUTHORITATIVE` (Drew's settled answers; default the whole root) and `DREW_DOCS_EVIDENCE` (supporting material, ingested with `authoritative = false`). `DREW_DOCS_SKIP` leaves paths out even inside a layer. The longest listed path containing a file decides its layer; files under no listed path are not ingested. This keeps answer keys, drafts and working notes out of the corpus.

Ingestion:

- `drew ingest` reads the full snapshot under `DREW_DOCS_DIR` on each run. Files removed from the directory have their current chunks superseded.
- Each chunk records, even though v1 does not enforce them: ACL (empty for markdown), customer (if any), sensitivity (`public` / `internal` / `confidential`).
- Chunks are **immutable and content-addressed**. Re-ingestion inserts new versions and marks old ones superseded, so traces stay reproducible.
- Every chunk records the embedding model and version. Changing models means a full re-embed; mixing is never allowed.
- Ingested text is treated as data: delimited in prompts, never as instructions.

## 3. Stack

| Concern | Choice |
|---|---|
| Language | Rust, single crate, one binary: `drew serve`, `drew ingest`, `drew eval`, `drew models check` |
| Modules | `ingest`, `retrieval`, `pipeline`, `llm`, `server`, `slack`, `eval` |
| Store | Postgres + pgvector (Docker Compose for the demo; Aurora later, where native RLS covers the security phase) |
| Retrieval | Hybrid: full-text + vector, merged with reciprocal rank fusion (SAP identifiers like `S_TCODE`, `AGR_USERS` embed poorly). Full text ranks exact identifier matches first, then BM25 in plain SQL over corpus statistics that `drew ingest` refreshes (no extension, so it runs on Aurora). Before searching, a question gains the words of the glossary table rows it names by a part, alternative or acronym of their term (`retrieval.expansion` lists the glossary documents; parsed from their current chunks on each retrieval, capped per question, recorded in the `Retrieve` trace) |
| Embeddings, "why" LLM | Amazon Bedrock (AI Council approved), dev AWS account. Otherwise any OpenAI-compatible server: locally Ollama or llama.cpp, e.g. `bge-m3` and `qwen3:14b`; hosted, Cloudflare Workers AI, e.g. `@cf/baai/bge-m3` and `@cf/qwen/qwen3.8-27b`. Each port names its backend (`DREW_EMBED_BACKEND`, `DREW_EXPLAIN_BACKEND`), with no default |
| System One | TypeSafe's System One API (`POST /v1/systemone`): yes/no, choice, and score questions over a text, answered with calibrated probabilities. Served locally by [Kev](https://github.com/jaredpalmer/kev) (MLX on Apple Silicon) and hosted by system-one.dev; Cloudflare's Clef models answer the same request on Workers AI behind their own route and envelope. `DREW_SYSTEM_ONE_BACKEND` is `none`, `system-one` or `workers-ai` |
| Libraries | `axum`, `sqlx`, `pgvector`, `aws-sdk-bedrockruntime`, `reqwest`, `serde` |

## 4. API

OpenAI-compatible:

- `GET /v1/models` → a single model, `drew-ai`.
- `POST /v1/chat/completions`
  - Only the last `user` message is used. Client `system` messages and history are ignored.
  - `stream: true` is accepted; the verified answer is returned as a single chunk (citations must be verified before anything is shown).
  - Auth: per-user bearer key, mapped to a user id in the run record.
- `GET /v1/runs/{id}` → a persisted run with its trace.

Response:

- `choices[0].message.content` — markdown answer in the per-category format, with numbered citations.
- Top-level `drew` object — category, outcome (`answered` / `declined`), optional halt metadata, verdict, and citations (chunk id, version, title, URL, excerpt). **No trace or per-step scores** in the chat response; those are audit data on `GET /v1/runs/{id}`.
- `DREW_ADMIN_USERS` may read any persisted run (for example Slack runs stored as `slack:{user_id}`).

Per-category format:

| Category | Content |
|---|---|
| `boolean` | **Yes** / **No**, confidence, the why, citations |
| `choice` | Selected option, one line per rejected option, citations |
| `ranking` | Ordered list, one citation per item, then the why |
| `open` | LLM answer, citation per claim |
| declined | Fixed "Can't answer with certainty" message + ranked sources |

Failures return 5xx with the run id.

## 5. Pipeline

### 5.1 Model

A typestate state machine with a Writer-style trace that short-circuits on halt:

- **Typestate**: `Run<TCategory, TState>`. A step can only be applied when its input type equals the current state, so invalid sequences do not compile.
- **Writer**: the `then` combinator times the step, appends its trace entry, and checks its threshold. Steps never see or mutate the trace.
- **Short-circuit**: a decline or failure produces `Halted`, which still carries the entire trace up to that point.

### 5.2 Core types

```rust
pub struct Unclassified;
pub struct Boolean;
pub struct Choice;
pub struct Ranking;
pub struct Open;

pub struct Run<TCategory, TState> {
    id: RunId,
    question: Question,
    state: TState,
    trace: Trace,
    category: PhantomData<TCategory>,
}

pub trait Step: Send + Sync {
    const NAME: &'static str;
    type Category: Category;
    type Input: Send;
    type Output: Send;
    type Record: Into<TraceRecord> + Send;

    fn threshold(&self) -> Option<Probability> { None }

    fn run(&self, question: &Question, input: Self::Input)
        -> impl Future<Output = Result<Scored<Self::Output, Self::Record>, StepError>> + Send;
}

impl<TCategory, TState> Run<TCategory, TState> {
    /// Times the step, appends its trace entry, checks its threshold, and continues or halts.
    pub async fn then<TStep>(self, step: &TStep) -> Result<Run<TCategory, TStep::Output>, Halted>
    where
        TStep: Step<Category = TCategory, Input = TState>;
}

pub struct Halted {
    id: RunId,
    question: Question,
    trace: Trace,
    category: Option<CategoryKind>, // None when the run halted before classification
    at: &'static str,
    halt: Halt,
}

pub enum Halt {
    Declined { reason: DeclineReason, score: Option<Probability>, threshold: Option<Probability> },
    Failed { error: String },
}
```

Steps are generic over the categories they support (`Answer<Boolean>` is a step; `Answer<Open>` is not), rather than `Step` being generic over the category. With `Step<TCategory>`, the blanket `Pipeline` impl below conflicts with the `Seq` impl, because another crate could implement `Step<TheirType>` for `Seq`.

Steps take their input by value and also see the question; they never see the trace. A step returns `StepError::decline(reason)` for a domain decline (for example invalid model output) and attaches its record so the raw output stays in the trace. Any other error is a failure.

### 5.3 Pipelines per category

| Category | Steps |
|---|---|
| `Boolean`, `Choice`, `Ranking` | `Classify` ∥ `Retrieve` → `Rerank` → `Answer` → `Explain` → `Verify` |
| `Open` | `Classify` ∥ `Retrieve` → `Rerank` → `Explain` → `Verify` |

A pipeline is a typed path through the same state machine. A `Step` is a one-transition pipeline, and `Seq` composes two pipelines only when the first one's output state is the second one's input state.

```rust
pub trait Pipeline: Send + Sync {
    type Category: Category;
    type Input: Send;
    type Output: Send;

    fn drive(&self, run: Run<Self::Category, Self::Input>)
        -> impl Future<Output = Result<Run<Self::Category, Self::Output>, Halted>> + Send;
}

impl<TStep: Step> Pipeline for TStep {
    type Category = TStep::Category;
    type Input = TStep::Input;
    type Output = TStep::Output;

    async fn drive(&self, run: Run<TStep::Category, TStep::Input>)
        -> Result<Run<TStep::Category, TStep::Output>, Halted>
    {
        run.then(self).await
    }
}

pub struct Seq<TFirst, TRest>(pub TFirst, pub TRest);

impl<TFirst, TRest> Pipeline for Seq<TFirst, TRest>
where
    TFirst: Pipeline,
    TRest: Pipeline<Category = TFirst::Category, Input = TFirst::Output>,
{
    type Category = TFirst::Category;
    type Input = TFirst::Input;
    type Output = TRest::Output;

    async fn drive(&self, run: Run<TFirst::Category, TFirst::Input>)
        -> Result<Run<TFirst::Category, TRest::Output>, Halted>
    {
        self.1.drive(self.0.drive(run).await?).await
    }
}
```

Pipelines are declared once, as types. Steps hold their model clients and thresholds, so pipeline values are built once at startup:

```rust
pub type Decisive<C> = Seq<Answer<C>, Seq<Explain<C>, Verify<C>>>; // Boolean, Choice, Ranking
pub type OpenEnded = Seq<Explain<Open>, Verify<Open>>;
```

`Decisive<C>` is a `Pipeline` only when every step in it implements `Step` for `C` and each step's output is the next step's input. There is no `impl Step for Answer<Open>`, so `Decisive<Open>` is not a `Pipeline`, and driving an `Open` run through it does not compile. There is no separate gate step: `Classify`, `Rerank`, and `Answer` are scored, and `then` checks their thresholds.

### 5.4 Classification and the parallel join

`Classify` and `Retrieve` run concurrently on `Run<Unclassified, Asked>` through `Run::both`, which traces both entries even when one branch halts (retrieval does not depend on the category). Their join produces the one place where runtime data becomes a type:

```rust
enum Classified {
    Boolean(Run<Boolean, Reranked>),
    Choice(Run<Choice, Reranked>),
    Ranking(Run<Ranking, Reranked>),
    Open(Run<Open, Reranked>),
}
```

Matched exhaustively, so adding a category does not compile until every arm is handled. If either parallel branch halts, the trace still contains the other branch's entry.

The join is the one transition that is not a `Pipeline`: it changes `TCategory` based on runtime data, so it is a `match`, and each arm hands off to its category's pipeline:

```rust
let answered = match classified {
    Classified::Boolean(run) => Answered::Boolean(self.boolean.drive(run).await?), // Decisive<Boolean>
    Classified::Choice(run) => Answered::Choice(self.choice.drive(run).await?),    // Decisive<Choice>
    Classified::Ranking(run) => Answered::Ranking(self.ranking.drive(run).await?), // Decisive<Ranking>
    Classified::Open(run) => Answered::Open(self.open.drive(run).await?),          // OpenEnded
};
```

Retrieval starting values (tuned by eval): top 50 hybrid chunks → expand to parent page/section/thread and dedupe → `Rerank` → top 5 to `Answer` and `Explain`.

Measured options for raising retrieval recall (BM25, query expansion, a grep channel, enrichment) are in [research/search/](research/search/README.md).

### 5.5 Answer and explain by category

| Category | `Answer` output | `Explain` input |
|---|---|---|
| `Boolean` | `Yes` / `No` | verdict + reranked sources |
| `Choice` | `Cited<Candidate>` chosen from `AtLeastTwo<Cited<Candidate>>` | verdict + reranked sources |
| `Ranking` | `Vec<Cited<Item>>` ordered by a criterion stated in the prompt | verdict + reranked sources |
| `Open` | — | reranked sources |

Rules:

- `Provenance` is `UserPrompt(Span)` or `Chunk(ChunkId)`. Every option, item, and claim is `Cited<T>`.
- A `choice` with fewer than 2 candidates, or a `ranking` without a stated criterion, is reclassified as `open`.
- Model output is deserialized into a raw DTO, then converted with `TryFrom` into domain types. Any conversion failure halts as a decline.
- The `Explain` LLM must flag when cited sources contradict the System One verdict.
- `Explain` applies the corpus's answering rules from its system prompt, never from retrieved text: authoritative sources outrank evidence; generation, store, vocabulary and scope are settled or named; no identifier that the sources lack; source provenance tags kept and the model's own inferences tagged `[INF]`; sandbox and tenant numbers never generalised; recorded unknowns stated, not guessed.
- `VerifiedAnswer` is only constructible inside `Verify`, which rejects any claim without a citation and any citation not among this run's retrieved chunks (runtime membership check).

### 5.6 Trace

Each step contributes one entry:

- Shared part: step name, parallel group, start time, duration, model id and version, prompt version, outcome.
- Step record (`Step::Record`): raw model output, parsed output, chunk ids and versions, score, threshold.

The trace is a flat list ordered by start time, serialized through the `TraceRecord` enum into a JSONB column in `pipeline_runs`. Answered, declined, and failed runs are all persisted. Chunk text is referenced by id and version, never copied.

## 6. Evaluation

- **Golden set**: about 50 questions per category, mined from Drew's Q&A threads, held out of the corpus, verified by Drew.
- **`drew eval`** reports per step: classification accuracy, retrieval recall@50, rerank hit@5, answer precision, decline rate, citation validity. Recall counts the matches retrieval ranked; the neighbouring sections it adds as context (`context_for` in the trace) neither count nor take a place.
- **`drew eval retrieval`** runs only `Retrieve` (embedder and database) and reports recall@10 and recall@50 per question and category, with each question gained, lost or unchanged against a baseline results file. Retrieval changes are gated here first: a lost question exits non-zero.
- **CI**: runs on every change to prompts, models, chunking, or thresholds; a regression in any metric fails the build.
- **Threshold tuning**: re-applies candidate thresholds to persisted runs from their recorded scores, without calling any model. Full replay is out of scope for v1. A candidate's `thresholds` and `per_category` have the same shape as those in `config/drew.json`, so a tuned candidate is pasted in as is.
- **Configuration**: every tunable (retrieval candidates and BM25 `k1`/`b`, `classify.min_confidence`, rerank `top_k`, `judge` (with its `candidates`) and `evidence_rank_penalty`, step thresholds) is in one JSON file, `config/drew.json` or `DREW_CONFIG`. Unknown keys are rejected. `drew eval run` prints it with the models in use and records both, with the git revision, in the results file; `models.rerank` is the model that judged Rerank, `null` when the judge is `none` or System One is not configured.
- **Launch gate** (opening beyond the 4 users): answered precision ≥ 95% and decline rate ≤ ~40%. Until then v1 is the shadow phase, with Drew reviewing.
- **Feedback loop**: declined and thumbs-down answers go to Drew's review queue. His corrections are ingested as authoritative chunks, and the question joins the golden set.

## 7. Milestones

1. Ingestion and storage, plus the golden set.
2. Hybrid retrieval, measured by recall@50.
3. `open` end to end as the baseline: `Retrieve` → `Explain` → `Verify`, with every question treated as `open` and `Rerank` stubbed as top-5 by fused score.
4. `Classify`: one System One choice question over the four categories, scored by its calibrated probability. Until `Answer` exists, `boolean`, `choice`, and `ranking` questions are answered through the `open` path, and the run keeps the classified category.
5. `Rerank`: keeps the best `top_k` matches; `rerank.judge` chooses who judges them, independently of `Classify` and `Answer`, which use System One either way.
   - `"none"` (the default): no model call. Matches are ordered by reciprocal rank fusion of their retrieval rank alone, with evidence (non-authoritative) matches ranked 10 places lower; the step is unscored, so no `rerank` threshold applies.
   - `{"system_one": {"candidates": 25}}`: one System One yes/no question per match ("does this text help answer the question?") over the top 25 matches by fused score. Matches are ordered by reciprocal rank fusion of their retrieval rank and their probability-of-yes rank, with the same evidence penalty in both; the best kept probability scores the step. `candidates` belongs to this variant, so a config cannot give candidates without a judge; it must be at least `top_k`. A flat `rerank.candidates` is rejected on load; results files written before `judge` existed read it as this variant.

   Ordering by probability alone scored hit@5 36% on the DaaS golden set against 53% for retrieval order, because Kev's probabilities sit close together. Fused with retrieval rank, on 2026-10-22, it cost about 40 s and 25 System One calls per question for hit@5 43/58 against 41/58 in retrieval order, and 7/15 against 10/15 on `boolean` questions, so the judge is `none`. Neighbouring sections added by retrieval are context for their match, not candidates of their own.
6. `Answer` with thresholds and declines. `Answer<Boolean>` is built: one System One yes/no question over the reranked sources; the probability of yes is the verdict, `max(p, 1 - p)` scores the step against `answer`'s threshold, and `Explain<Boolean>` declines the run (`ContradictsVerdict`) when the sources contradict the verdict. A `boolean`, `choice` or `ranking` classification below `classify.min_confidence` falls back to `open`. `choice` and `ranking` still take the `open` path.
7. Launch-gate evaluation.

## 8. Deferred: security phase

Designed, not built:

- Postgres RLS policies over the recorded labels; filter inside the vector query, recheck before LLM context.
- Users retrieve only what their groups may read (identity mapped by email, group memberships cached).
- `confidential` chunks restricted to a Drew-owned maintainers group.
- Customers see only chunks with their own customer label or none.
- Ingestion on change to the markdown directory, with revocation effective within 1 hour and freshness within 24 hours, plus a nightly permission resync.
- Slack bot hardening beyond v1 (channel allowlists, rate limits).
- EKS deployment through the shared GitOps repos, Aurora, SageMaker for System One.
- Audit log retention: 1 year.

## 9. Open items

- Confidence threshold values per step and category (set by `drew eval`).
- Which Bedrock embedding model (Titan vs Cohere Embed), chosen by retrieval recall in milestone 2.
- Which System One model: Kev size (0.8B, 4B, 9B, 27B), a Kev fine-tuned on the golden set, Laya, or a hosted Jev, chosen by milestone 4–6 metrics.
