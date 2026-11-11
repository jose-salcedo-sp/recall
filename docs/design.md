# recall — Design

recall is a backend-agnostic provenance system: it answers questions from an ingested corpus so that every claim carries the exact chunk versions it came from, is checked against them before it is shown, and is stored with a full trace. A judge (a small, fast calibrated model; System One ships as its backend) answers the **what** (yes/no, a choice, or a ranking), and admits sources and checks claims; an LLM answers the **why**. It ships as a library crate and one binary with an OpenAI-compatible HTTP API.

The core is domain-free. A domain application brings its corpus (a `Source` connector), its answering rules (`explain.instructions`), its glossary documents, golden set and thresholds (config), and any front end beyond the HTTP API. Confidence thresholds live in `config/recall.json`; the evaluation harness tunes them per corpus and model.

## 1. Scope

| Built | Deferred |
|---|---|
| OpenAI-compatible HTTP API | Other front ends (chat bots and the like): consumers of the API or the library, not core modules |
| Batch ingestion through the `Source` port; a markdown-directory connector | Event-driven ingestion, revocation SLAs |
| Admission and support checks behind the `Judge` port | Regeneration after a failed claim |
| Labels on every chunk, enforced by the `Authorizer` port at every stage (section 8) | Postgres RLS, per-question principals in eval |
| Dockerfile + env config | A deployment target |

Guiding constraints:

- **Provenance is mandatory.** No claim, option, or ranked item exists without a source.
- **The judge never invents.** Options and ranked items come only from the user prompt or retrieved chunks.
- **Nothing is shown before verification.** Every claim passes `Verify` first.
- **Everything is auditable.** Every run, including declines and failures, is persisted with a full trace.
- **Invalid pipelines do not compile.** Each category's pipeline contains only the steps it needs.
- **No vendor or domain in the core.** Models, stores and sources sit behind ports with fakes and contract tests.

## 2. Corpus and ingestion

The `Source` port returns a full snapshot of documents; `ingest` versions, embeds and stores them. A connector names itself with a `SourceName` (one or more of `a-z`, `0-9`, `_`, `-`), which prefixes its document ids (`{source}:{external_id}`) and scopes its snapshot: a snapshot supersedes only documents stored under its own name, so one ingestion rejects two connectors of one name. Neither the core nor the schema lists the names, so a domain adds a connector without editing recall: it implements `Source`, passes it to `ingest::ingest` (or, from a binary that reuses recall's command line, `cli::ingest::run_with`), and runs the port's contract suite (`recall::source_contract!`, `testing` feature) against it. The shipped connector reads a directory of markdown files:

- Each file is one document; each ATX heading section is one chunk. Optional `RECALL_DOCS_BASE_URL` turns paths into citation links.
- Two layers, chosen by path under the root: `RECALL_DOCS_AUTHORITATIVE` (settled answers; default the whole root) and `RECALL_DOCS_EVIDENCE` (supporting material, ingested with `authoritative = false`). `RECALL_DOCS_SKIP` leaves paths out even inside a layer. The longest listed path containing a file decides its layer; files under no listed path are not ingested. This keeps answer keys, drafts and working notes out of the corpus.

Ingestion:

- `recall ingest` reads the full snapshot on each run. Documents no longer in the snapshot have their current chunks superseded.
- Each chunk records labels: ACL (empty for markdown) and sensitivity (`public` < `internal` < `confidential`). An ACL entry is tagged, `user:<id>` or `role:<name>`, so a user id never matches a role. The `Labeler` port sets them per document before versioning; the shipped `LabelRules` tightens the connector's labels by source and path prefix from the access policy (ACLs intersected, the higher sensitivity), so a rule never widens access, and a document it cannot label is not stored.
- Chunks are **immutable and content-addressed**. Re-ingestion inserts new versions and marks old ones superseded, so traces stay reproducible.
- Every chunk records the embedding model and version. Changing models means a full re-embed; mixing is never allowed.
- Ingested text is treated as data: delimited in prompts, never as instructions.

## 3. Stack

| Concern | Choice |
|---|---|
| Language | Rust, one library crate and one binary: `recall serve`, `recall ingest`, `recall eval`, `recall models check` |
| Modules | `ingest`, `retrieval`, `pipeline`, `llm`, `store`, `server`, `eval` |
| Store | Postgres + pgvector, behind the `ChunkStore` and `RunStore` ports; time and run ids behind `Clock` and `RunIds`, eval's files behind `Files` |
| Retrieval | Hybrid: full-text + vector, merged with reciprocal rank fusion (identifiers like `A_OPCODE` or `ROLE_USERS` embed poorly). Full text ranks exact identifier matches first, then BM25 in plain SQL over corpus statistics that `recall ingest` refreshes (no extension). Before searching, a question can gain the words of the glossary table rows it names by a part, alternative or acronym of their term (`retrieval.expansion.documents` lists the glossary documents; parsed from their current chunks on each retrieval, capped per question, recorded in the `Retrieve` trace) |
| Embeddings, "why" LLM | Adapters for Amazon Bedrock and any OpenAI-compatible server (locally Ollama or llama.cpp; hosted, Cloudflare Workers AI). Each port names its backend (`RECALL_EMBED_BACKEND`, `RECALL_EXPLAIN_BACKEND`), with no default |
| System One | Adapters for the System One API (`POST /v1/systemone`: yes/no, choice, and score questions over a text, answered with calibrated probabilities), served locally by [Kev](https://github.com/jaredpalmer/kev) and hosted by system-one.dev, and for Cloudflare's Clef models on Workers AI. `RECALL_SYSTEM_ONE_BACKEND` is `none`, `system-one` or `workers-ai` |
| Judge | The `Judge` port; `SystemOneJudge` adapts whichever System One is configured |
| Libraries | `axum`, `sqlx`, `pgvector`, `aws-sdk-bedrockruntime`, `reqwest`, `serde` |

## 4. API

OpenAI-compatible:

- `GET /v1/models` → a single model, `recall`.
- `POST /v1/chat/completions`
  - Only the last `user` message is used. Client `system` messages and history are ignored.
  - `stream: true` is accepted; the verified answer is returned as a single chunk (citations must be verified before anything is shown).
  - Auth: per-user bearer key, mapped to a user id in the run record and to the run's principal (section 8).
- `GET /v1/runs/{id}` → a persisted run with its trace.

Response:

- `choices[0].message.content` — markdown answer in the per-category format, with numbered citations.
- Top-level `provenance` object — run id, category, outcome (`answered` / `declined` / `failed`), optional halt metadata, verdict, and citations (chunk id, version, title, URL, excerpt). **No trace or per-step scores** in the chat response; those are audit data on `GET /v1/runs/{id}`.
- The access policy decides who reads a persisted run (`runs`: `own` or `any`); a run the reader may not read is a 404. A reader of its own runs only does not see denied access checks.

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

    fn run(&self, ask: &Ask, input: Self::Input) // question, principal, access
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

Steps take their input by value and also see the `Ask`: the question, who asked, and what they may read; they never see the trace. `Run::authorize` builds it from the `Authorizer` and records it as the run's first trace entry. A step returns `StepError::decline(reason)` for a domain decline (for example invalid model output) and attaches its record so the raw output stays in the trace. Any other error is a failure.

### 5.3 Pipelines per category

| Category | Steps |
|---|---|
| `Boolean`, `Choice`, `Ranking` | `Classify` ∥ `Retrieve` → `Rerank` [→ `Admit`] → `Answer` → `Explain` → `Verify` |
| `Open` | `Classify` ∥ `Retrieve` → `Rerank` [→ `Admit`] → `Explain` → `Verify` |

`Admit` is in a run only when `admit.judge` names a judge; it maps `Reranked` to `Reranked`, so the pipeline types after it do not change. The boolean path exists only when a judge classifies (5.8).

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
enum Classified<'a> {
    Boolean(Run<Boolean, Reranked>, &'a Decisive<Boolean>), // assembled only when a judge classifies
    Choice(Run<Choice, Reranked>),
    Ranking(Run<Ranking, Reranked>),
    Open(Run<Open, Reranked>),
}
```

Matched exhaustively, so adding a category does not compile until every arm is handled. If either parallel branch halts, the trace still contains the other branch's entry.

The join is the one transition that is not a `Pipeline`: it changes `TCategory` based on runtime data, so it is a `match`, and each arm hands off to its category's pipeline:

```rust
let answered = match classified {
    Classified::Boolean(run, boolean) => Answered::Boolean(boolean.drive(run).await?), // Decisive<Boolean>
    Classified::Choice(run) => Answered::Choice(self.choice.drive(run).await?),    // Decisive<Choice>
    Classified::Ranking(run) => Answered::Ranking(self.ranking.drive(run).await?), // Decisive<Ranking>
    Classified::Open(run) => Answered::Open(self.open.drive(run).await?),          // OpenEnded
};
```

Retrieval starting values (tuned by eval): top 50 hybrid chunks → expand to parent page/section/thread and dedupe → `Rerank` → top 5 to `Answer` and `Explain`.

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
- The `Explain` LLM must flag when cited sources contradict the judge's verdict.
- `Explain` applies answering rules from its system prompt, never from retrieved text. The generic rules are fixed: authoritative sources outrank evidence; an unsettled dependency is answered per case or named; no identifier the sources lack; quoted text is verbatim; the model's own inferences are tagged `[INF]`; recorded unknowns are stated, not guessed. A domain adds its own rules in `explain.instructions` (trusted config), after the generic ones, which it cannot relax.
- `VerifiedAnswer` is only constructible inside `Verify`, which rejects any claim without a citation and any citation not among this run's retrieved chunks (runtime membership check), then applies the configured support check (5.7).

### 5.7 Provenance checks

Every judged step (`Classify`, `Rerank`, `Answer`, `Admit`, `Verify`) asks through `llm::judge::Judge`, a vendor-neutral port (yes/no and one-of questions over a text, answered with probabilities); `SystemOneJudge` adapts any System One backend, so no step builds a wire request. `clippy.toml` disallows System One's `DecisionRequest` outside `src/llm/`. Config modes name the role (`judged`, or the judge's settings), never the backend.

- **Admission** (`Admit`, `admit.judge`): each reranked match is asked whether it tries to change the assistant's instructions, addresses the question, contradicts its premise, and states usable evidence. It is routed by the first cutoff it crosses: injection and irrelevant matches are excluded, a contradicting match is kept as a conflict, and the rest are kept only with evidence. Context is screened for injection only and otherwise follows its match. No kept match declines with `NoSources`. The record lists every source's probabilities and route.
- **Support** (`Verify`, `verify.support`): `membership` checks citations only; `quotes` also requires every span in backticks or double quotes to appear in a cited chunk (`Explain`'s prompt says such spans are verbatim); `{"judged": {"min_support": 0.75}}` adds a one-of question per cited chunk (supports, contradicts, says nothing), in citation order, until one supports the claim with at least `min_support`. The chosen option's own probability is compared, not an entropy confidence. Without a judge, `judged` does not assemble (5.8); the record says which check ran.
- **Publishing** (`verify.publish`): `decline` declines on any failed claim; `strip` leaves failed claims out and declines only when none is left, so what is shown is still verified.
- **Audit**: the Admit and Verify records, with every probability and verdict, are part of the trace in `pipeline_runs`; `recall eval` reports `faithfulness`, the share of judged claims that are supported.

Not built: regenerating once after a failed claim (a loop back to `Explain` that the typed pipeline has no combinator for), visibility rules over the recorded labels (section 8), and streaming stage events (nothing is shown before verification).

### 5.6 Trace

Each step contributes one entry:

- Shared part: step name, parallel group, start time, duration, model id and version, prompt version, outcome, and one access decision per chunk the step checked.
- Step record (`Step::Record`): raw model output, parsed output, chunk ids and versions, score, threshold.

The trace is a flat list ordered by start time, serialized through the `TraceRecord` enum into a JSONB column in `pipeline_runs`. Answered, declined, and failed runs are all persisted. Chunk text is referenced by id and version, never copied.

### 5.8 Assembly

`pipeline::assembly` resolves the config against the model backends available (`Ports`: retriever, Explain LLM, optional `Judge`) once, before any question, and builds every step. It is the only place that decides what an absent judge means:

- A check the config names runs as named, or assembly fails with `Unassembled`: `rerank.judge`, `admit.judge` or `verify.support` naming a judge with none configured lists every such key in one error, and a `rerank` threshold while `rerank.judge` is `none` (the step is unscored) is an error too. No step falls back to a quieter check.
- Without a judge, `Classify` records every question as `open` and there is no boolean path: no config key asks for classification, so this follows from the backends rather than downgrading a check.
- Steps receive resolved modes (`RerankBy`, `Support`, `Option<Admit>`), never an optional client to test at run time.

What was assembled is a `Plan` (the classify, rerank, admit and support modes), logged by `recall serve` at startup and recorded by `recall eval run`. Each run's Classify, Rerank and Verify trace records also carry the mode they ran in, and a run has an `admit` entry only when admission is on.

## 6. Evaluation

- **Golden set**: JSON lines of questions with their category, expected answer and relevant chunk ids, held out of the corpus. A domain supplies its own; `eval/golden/example.jsonl` is a synthetic example of the format.
- **`recall eval`** reports per step: classification accuracy, retrieval recall@50, rerank hit@5, answer precision, decline rate, citation validity, and faithfulness (supported claims among those a judge checked). Recall counts the matches retrieval ranked; the neighbouring sections it adds as context (`context_for` in the trace) neither count nor take a place.
- **`recall eval retrieval`** runs only `Retrieve` (embedder and database) and reports recall@10 and recall@50 per question and category, with each question gained, lost or unchanged against a baseline results file. Retrieval changes are gated here first: a lost question exits non-zero.
- **Baseline gate**: `--baseline` fails a run when any metric regresses past its tolerance, overall or per category.
- **Threshold tuning**: re-applies candidate thresholds to persisted runs from their recorded scores, without calling any model. A candidate's `thresholds` and `per_category` have the same shape as those in `config/recall.json`, so a tuned candidate is pasted in as is.
- **Configuration**: every tunable (retrieval candidates, BM25 `k1`/`b` and glossary expansion, `classify.min_confidence`, rerank `top_k`, `judge` (with its `candidates`) and `evidence_rank_penalty`, the `admit` judge and cutoffs, `explain.max_tokens` and `explain.instructions`, `verify.support` and `verify.publish`, step thresholds) is in one JSON file, `config/recall.json` or `RECALL_CONFIG`. Unknown keys are rejected. `recall eval run` prints it with the models in use and the assembled plan, and records all three, with the git revision, in the results file; `models.rerank`, `models.admit` and `models.verify` name the judging models, `null` when none judged.

## 7. Steps and their defaults

- `Rerank`: keeps the best `top_k` matches; `rerank.judge` chooses whether the judge orders them, independently of `Classify` and `Answer`, which use the judge whenever one is configured.
  - `"none"` (the default): no model call. Matches are ordered by reciprocal rank fusion of their retrieval rank alone, with evidence (non-authoritative) matches ranked 10 places lower; the step is unscored, so no `rerank` threshold applies.
  - `{"candidates": 25}`: one yes/no question to the judge per match ("does this text help answer the question?") over the top 25 matches by fused score. Matches are ordered by reciprocal rank fusion of their retrieval rank and their probability-of-yes rank, with the same evidence penalty in both; the best kept probability scores the step. `candidates` belongs to this variant, so a config cannot give candidates without a judge; it must be at least `top_k`. Neighbouring sections added by retrieval are context for their match, not candidates of their own.
- `Answer<Boolean>`: one yes/no question to the judge over the reranked sources; the probability of yes is the verdict, `max(p, 1 - p)` scores the step against `answer`'s threshold, and `Explain<Boolean>` declines the run (`ContradictsVerdict`) when the sources contradict the verdict. A `boolean`, `choice` or `ranking` classification below `classify.min_confidence` falls back to `open`. `choice` and `ranking` take the `open` path until their `Answer` steps exist.

## 8. Access control

Every run is asked by a `Principal` (a user id and the roles its authenticator asserts). The `Authorizer` port, injected in `Ports`, compiles it once per run into `Access`: a `ChunkScope` (the principal's ACL members and one grant per role, each a sensitivity clearance) and a `RunAccess`. A chunk is readable when its ACL is empty or names a member (`user:<the principal's id>` or `role:<one of its roles>`), and a grant clears its sensitivity. recall is single-tenant: no label or grant names a tenant. Deny by default: an unknown principal gets nothing, and an authorizer error fails the run at `authorize`, before retrieval.

- **Retrieval** filters by the scope in SQL: glossary, full-text, vector (HNSW iterative scan) and neighbour queries. The `Retriever` contract checks that exactly the readable chunks come back.
- **Every step that reads chunks** (`Retrieve`, `Rerank`, `Admit`, `Answer`, `Explain`, `Verify`) rechecks its sources against the same scope and records one decision per chunk, so a retriever that leaks still leaks nothing into a prompt, a citation or a decline.
- **Ingestion** labels through the `Labeler` port; **eval** asks as an explicit `--as` principal.
- `RulePolicy` (roles, `everyone`, `members`, label rules from `config/access.json`) is the shipped adapter; `access::contract` holds the port's rules for other adapters. `everyone` is a baseline grant: every authenticated caller holds those roles, known to the policy or not, so a non-empty `everyone` opts out of deny by default. The shipped policy leaves it empty and lists `admin` as a `maintainer`.

Not built: Postgres RLS, ingestion on change with bounded revocation delays, and per-question principals in eval.

## 9. Open items

- Confidence thresholds and admission and support cutoffs per corpus and model (set by `recall eval`).
- `choice` and `ranking` `Answer` steps.
