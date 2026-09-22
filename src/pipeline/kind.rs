use std::sync::Arc;

use crate::ctx::Ctx;
use crate::error::Result;
use crate::types::Stage;

/// Round 1: one Choice over the question. Kind is a switch, not a rewrite.
pub async fn run(ctx: &Arc<Ctx>, ask_id: uuid::Uuid, question: &str) -> Result<(String, f64)> {
    let k = ctx
        .with_retry(Stage::Kind, ctx.cfg.admit_timeout, || {
            ctx.system_one.kind(question)
        })
        .await?;
    let fell_back = k.confidence < ctx.cfg.kind_confidence_min;
    let model_choice = k.choice.clone();
    let (choice, confidence) = if fell_back {
        ("atomic_lookup".to_string(), k.confidence)
    } else {
        (k.choice, k.confidence)
    };
    tracing::info!(
        ask_id = %ask_id,
        kind = %choice,
        confidence,
        model_choice = %model_choice,
        fell_back,
        chitchat = choice == "chitchat",
        "kind chosen"
    );
    Ok((choice, confidence))
}

pub fn is_chitchat(kind: &str, confidence: f64, min: f64) -> bool {
    kind == "chitchat" && confidence >= min
}
