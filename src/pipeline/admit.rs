use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::ctx::Ctx;
use crate::error::Result;
use crate::types::{Candidate, Citation, Stage};

/// Admission. The stage the product exists for.
///
/// Retrieval returns passages that look right; this asks whether each one actually
/// answers the question. Semantic similarity is not the same as containing the answer,
/// and that gap is exactly what admission closes.
///
/// One batched call covers every candidate (plan.md Finding 3). On failure the rule
/// from architecture.md is absolute: retry once, then error or admit nothing. There is
/// no fallback that forwards unfiltered candidates to the generator, because that
/// would undo the product.
/// What admission decided, including the scores of everything it rejected.
///
/// The rejected scores are not incidental: a calibration fit needs the negatives, and
/// `GET /v1/asks/{id}` is where they are read from. Returning only the admitted
/// citations would leave the audit trail unable to do the job it exists for.
pub struct Admission {
    pub admitted: Vec<Citation>,
    pub scores: HashMap<Uuid, f64>,
}

pub async fn run(
    ctx: &Arc<Ctx>,
    question: &str,
    as_of: DateTime<Utc>,
    candidates: &[Candidate],
) -> Result<Admission> {
    if candidates.is_empty() {
        return Ok(Admission {
            admitted: Vec::new(),
            scores: HashMap::new(),
        });
    }

    let scores = ctx
        .with_retry(Stage::Admit, ctx.cfg.admit_timeout, || {
            ctx.system_one.score(question, as_of, candidates)
        })
        .await?;

    let mut scored: Vec<(&Candidate, f64)> = candidates
        .iter()
        .map(|c| {
            let noul = scores.get(&c.id.to_string()).copied().unwrap_or(0.0);
            (c, noul)
        })
        .collect();

    // Highest noul first, so truncating to max_citations keeps the best evidence.
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));

    let threshold = ctx.cfg.admit_threshold;
    let admitted: Vec<Citation> = scored
        .iter()
        .filter(|(_, noul)| *noul >= threshold)
        .take(ctx.cfg.max_citations)
        .enumerate()
        .map(|(index, (c, noul))| Citation {
            id: c.id,
            index,
            statement: c.statement.clone(),
            noul: *noul,
            origin: c.origin.clone(),
            grantor_name: c.grantor_name.clone(),
            text: c.text.clone(),
        })
        .collect();

    tracing::info!(
        candidates = candidates.len(),
        admitted = admitted.len(),
        threshold,
        calibrated = ctx.cfg.admit_threshold_is_calibrated,
        top_noul = scored.first().map(|(_, n)| *n),
        "admission complete"
    );

    // An empty admission where the best candidate was close to the line is the
    // signature of a badly set threshold, which is worth saying out loud while the
    // threshold is still uncalibrated.
    if admitted.is_empty() {
        if let Some((_, top)) = scored.first() {
            tracing::warn!(
                top_noul = top,
                threshold,
                "nothing admitted; best candidate scored below the threshold"
            );
        }
    }

    Ok(Admission {
        admitted,
        scores: candidates
            .iter()
            .map(|c| (c.id, scores.get(&c.id.to_string()).copied().unwrap_or(0.0)))
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn candidate(n: &str) -> Candidate {
        Candidate {
            id: Uuid::new_v4(),
            statement: n.to_string(),
            text: n.to_string(),
            origin: "personal".into(),
            grantor_name: None,
            rrf_score: 0.0,
            noul: None,
        }
    }

    /// The selection logic in isolation: threshold, ordering, and the citation cap.
    fn select(scored: Vec<(Candidate, f64)>, threshold: f64, max: usize) -> Vec<Citation> {
        let mut s = scored;
        s.sort_by(|a, b| b.1.total_cmp(&a.1));
        s.iter()
            .filter(|(_, n)| *n >= threshold)
            .take(max)
            .enumerate()
            .map(|(index, (c, noul))| Citation {
                id: c.id,
                index,
                statement: c.statement.clone(),
                noul: *noul,
                origin: c.origin.clone(),
                grantor_name: None,
                text: c.text.clone(),
            })
            .collect()
    }

    #[test]
    fn admits_only_above_threshold_best_first_and_caps() {
        let scored = vec![
            (candidate("weak"), 0.2),
            (candidate("strong"), 0.95),
            (candidate("mid"), 0.75),
            (candidate("also_strong"), 0.9),
        ];
        let out = select(scored, 0.7, 2);

        assert_eq!(out.len(), 2, "must respect max_citations");
        assert_eq!(out[0].statement, "strong", "highest noul first");
        assert_eq!(out[1].statement, "also_strong");
        assert_eq!(out[0].index, 0);
        assert_eq!(out[1].index, 1, "index must be contiguous for [memory_N]");
    }

    #[test]
    fn nothing_above_threshold_yields_empty_admit() {
        let scored = vec![(candidate("a"), 0.3), (candidate("b"), 0.69)];
        assert!(
            select(scored, 0.7, 4).is_empty(),
            "an empty admission is the correct answer, never a fallback to all candidates"
        );
    }
}
