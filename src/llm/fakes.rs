//! Deterministic stand-ins for tests and for running without AWS or a System One server.

// `ScriptedSystemOne` stands in for a System One backend, so it speaks the wire format the fence keeps in `src/llm/`.
#![allow(clippy::disallowed_types)]

use std::{collections::VecDeque, sync::Mutex};

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use super::{
    ChatModel, ChatRequest, ChatResponse, EMBEDDING_DIM, EmbedPurpose, Embedder, ModelRef,
    judge::{Judge, JudgeQuestion, JudgeRequest, Judgements, RawJudgement},
    system_one::{DecisionRequest, DecisionResponse, SystemOne},
};

/// Hashes each lowercase word into a bucket, so texts sharing words have a high cosine similarity.
pub struct HashEmbedder;

#[async_trait]
impl Embedder for HashEmbedder {
    fn model(&self) -> ModelRef {
        ModelRef::new("fake-hash-embedder", "1")
    }

    async fn embed(&self, texts: &[String], _purpose: EmbedPurpose) -> eros::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|text| hash_embed(text)).collect())
    }
}

fn hash_embed(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0; EMBEDDING_DIM];
    for word in text.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if word.is_empty() {
            continue;
        }
        let digest = Sha256::digest(word.to_lowercase().as_bytes());
        let bucket = u16::from_be_bytes([digest[0], digest[1]]) as usize % EMBEDDING_DIM;
        vector[bucket] += 1.0;
    }
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vector.iter_mut().for_each(|x| *x /= norm);
    }
    vector
}

/// [`HashEmbedder`] vectors under any model name, recording every batch it embeds.
pub struct RecordingEmbedder {
    model: ModelRef,
    batches: Mutex<Vec<Vec<String>>>,
}

impl RecordingEmbedder {
    /// Records under `model`, so a test can stand for an embedder other than [`HashEmbedder`].
    pub fn named(model: ModelRef) -> Self {
        Self {
            model,
            batches: Mutex::new(Vec::new()),
        }
    }

    /// Every batch embedded so far, in order.
    pub fn batches(&self) -> Vec<Vec<String>> {
        self.batches
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .clone()
    }
}

/// Records under [`HashEmbedder`]'s model.
impl Default for RecordingEmbedder {
    fn default() -> Self {
        Self::named(HashEmbedder.model())
    }
}

#[async_trait]
impl Embedder for RecordingEmbedder {
    fn model(&self) -> ModelRef {
        self.model.clone()
    }

    async fn embed(&self, texts: &[String], purpose: EmbedPurpose) -> eros::Result<Vec<Vec<f32>>> {
        self.batches
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .push(texts.to_vec());
        HashEmbedder.embed(texts, purpose).await
    }
}

/// Fails every call to every model port, as an unreachable backend would, for error-path tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct FailingModel;

const UNREACHABLE: &str =
    "error sending request for url (http://models.internal/v1): connection refused";

#[async_trait]
impl Embedder for FailingModel {
    fn model(&self) -> ModelRef {
        ModelRef::new("fake-failing", "1")
    }

    async fn embed(&self, _: &[String], _: EmbedPurpose) -> eros::Result<Vec<Vec<f32>>> {
        eros::bail!("{UNREACHABLE}")
    }
}

#[async_trait]
impl ChatModel for FailingModel {
    fn model(&self) -> ModelRef {
        ModelRef::new("fake-failing", "1")
    }

    async fn complete(&self, _: ChatRequest) -> eros::Result<ChatResponse> {
        eros::bail!("{UNREACHABLE}")
    }
}

#[async_trait]
impl SystemOne for FailingModel {
    fn model(&self) -> ModelRef {
        ModelRef::new("fake-failing", "1")
    }

    async fn decide(&self, _: DecisionRequest) -> eros::Result<DecisionResponse> {
        eros::bail!("{UNREACHABLE}")
    }
}

#[async_trait]
impl Judge for FailingModel {
    fn model(&self) -> ModelRef {
        ModelRef::new("fake-failing", "1")
    }

    async fn judge(&self, _: JudgeRequest) -> eros::Result<Judgements> {
        eros::bail!("{UNREACHABLE}")
    }
}

/// Returns queued responses in order and records every request it received.
pub struct ScriptedChat {
    model: ModelRef,
    responses: Mutex<VecDeque<eros::Result<ChatResponse>>>,
    requests: Mutex<Vec<ChatRequest>>,
}

impl ScriptedChat {
    /// Each call to `complete` takes the next response; once they run out, it fails.
    pub fn new(responses: impl IntoIterator<Item = eros::Result<ChatResponse>>) -> Self {
        Self {
            model: ModelRef::new("fake-scripted-chat", "1"),
            responses: Mutex::new(responses.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    /// [`ScriptedChat::new`] with successful replies of these texts.
    pub fn replying(texts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::new(texts.into_iter().map(|text| {
            Ok(ChatResponse {
                text: text.into(),
                mean_logprob: None,
            })
        }))
    }

    /// Every request received so far, in order.
    pub fn requests(&self) -> Vec<ChatRequest> {
        self.requests
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .clone()
    }
}

#[async_trait]
impl ChatModel for ScriptedChat {
    fn model(&self) -> ModelRef {
        self.model.clone()
    }

    async fn complete(&self, request: ChatRequest) -> eros::Result<ChatResponse> {
        self.requests
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .push(request);
        self.responses
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .pop_front()
            .unwrap_or_else(|| Err(eros::error!("ScriptedChat has no responses left")))
    }
}

/// Returns queued responses in order and records every request it received, under its own model name as
/// `HttpSystemOne` sends it.
pub struct ScriptedSystemOne {
    responses: Mutex<VecDeque<eros::Result<DecisionResponse>>>,
    requests: Mutex<Vec<DecisionRequest>>,
}

impl ScriptedSystemOne {
    /// Each call to `decide` takes the next response; once they run out, it fails.
    pub fn new(responses: impl IntoIterator<Item = eros::Result<DecisionResponse>>) -> Self {
        Self {
            responses: Mutex::new(responses.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    /// Every request received so far, in order, with `model` set as sent.
    pub fn requests(&self) -> Vec<DecisionRequest> {
        self.requests
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .clone()
    }
}

#[async_trait]
impl SystemOne for ScriptedSystemOne {
    fn model(&self) -> ModelRef {
        ModelRef::new("fake-system-one", "1")
    }

    async fn decide(&self, request: DecisionRequest) -> eros::Result<DecisionResponse> {
        self.requests
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .push(DecisionRequest {
                model: self.model().id,
                ..request
            });
        self.responses
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .pop_front()
            .unwrap_or_else(|| Err(eros::error!("ScriptedSystemOne has no responses left")))
    }
}

type Answer = dyn Fn(&str, &JudgeQuestion) -> RawJudgement + Send + Sync;

/// Answers each question with a function of the text and the question, and records every request it received.
pub struct FnJudge {
    answer: Box<Answer>,
    requests: Mutex<Vec<JudgeRequest>>,
}

impl FnJudge {
    /// `answer` is called once per question, in order.
    pub fn new(
        answer: impl Fn(&str, &JudgeQuestion) -> RawJudgement + Send + Sync + 'static,
    ) -> Self {
        Self {
            answer: Box::new(answer),
            requests: Mutex::new(Vec::new()),
        }
    }

    /// Every request received so far, in order.
    pub fn requests(&self) -> Vec<JudgeRequest> {
        self.requests
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .clone()
    }
}

#[async_trait]
impl Judge for FnJudge {
    fn model(&self) -> ModelRef {
        ModelRef::new("fake-fn-judge", "1")
    }

    async fn judge(&self, request: JudgeRequest) -> eros::Result<Judgements> {
        let answers = request
            .questions
            .iter()
            .map(|question| (self.answer)(&request.text, question))
            .collect();
        self.requests
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .push(request);
        Ok(Judgements::new(answers))
    }
}
