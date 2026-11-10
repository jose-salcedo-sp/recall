//! Deterministic stand-ins for tests and for running without AWS or a System One server.

use std::{collections::VecDeque, sync::Mutex};

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use super::{
    ChatModel, ChatRequest, ChatResponse, EMBEDDING_DIM, EmbedPurpose, Embedder, ModelRef,
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
