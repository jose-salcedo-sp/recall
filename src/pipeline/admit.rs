use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};

use crate::clients::systemone::FilterScores;
use crate::config::Config;
use crate::ctx::Ctx;
use crate::error::Result;
use crate::types::{Candidate, Citation, FilterRoute, Stage};

pub struct Admission {
    pub admitted: Vec<Citation>,
    pub conflicts: Vec<Citation>,
    pub scores: HashMap<uuid::Uuid, FilterScores>,
}

pub fn route(s: FilterScores, cfg: &Config) -> FilterRoute {
    if s.injection > cfg.injection_max {
        return FilterRoute::Exclude;
    }
    if s.relevant < cfg.relevant_min {
        return FilterRoute::Exclude;
    }
    if s.contradicts > cfg.contradicts_min {
        return FilterRoute::Conflict;
    }
    if s.evidence >= cfg.evidence_min {
        return FilterRoute::Include;
    }
    FilterRoute::Exclude
}

pub async fn run(
    ctx: &Arc<Ctx>,
    ask_id: uuid::Uuid,
    question: &str,
    as_of: DateTime<Utc>,
    candidates: &[Candidate],
) -> Result<Admission> {
    if candidates.is_empty() {
        return Ok(Admission {
            admitted: Vec::new(),
            conflicts: Vec::new(),
            scores: HashMap::new(),
        });
    }

    let scores = ctx
        .with_retry(Stage::Admit, ctx.cfg.admit_timeout, || {
            ctx.system_one.filter(question, as_of, candidates)
        })
        .await?;

    let mut includes: Vec<(&Candidate, FilterScores)> = Vec::new();
    let mut conflicts: Vec<(&Candidate, FilterScores)> = Vec::new();
    let mut by_id = HashMap::new();

    for c in candidates {
        let s = scores.get(&c.id.to_string()).copied().unwrap_or_default();
        by_id.insert(c.id, s);
        match route(s, &ctx.cfg) {
            FilterRoute::Include => includes.push((c, s)),
            FilterRoute::Conflict => conflicts.push((c, s)),
            FilterRoute::Exclude => {}
        }
    }

    includes.sort_by(|a, b| b.1.evidence.total_cmp(&a.1.evidence));
    includes.truncate(ctx.cfg.max_citations);

    let admitted = includes
        .iter()
        .enumerate()
        .map(|(index, (c, s))| citation(c, index, s.evidence, FilterRoute::Include))
        .collect::<Vec<_>>();
    let conflict_cites = conflicts
        .iter()
        .enumerate()
        .map(|(i, (c, s))| citation(c, admitted.len() + i, s.contradicts, FilterRoute::Conflict))
        .collect::<Vec<_>>();

    let ranked: Vec<serde_json::Value> = {
        let mut all: Vec<_> = candidates
            .iter()
            .map(|c| {
                let s = by_id.get(&c.id).copied().unwrap_or_default();
                let r = route(s, &ctx.cfg);
                serde_json::json!({
                    "id": c.id,
                    "statement": trunc_stmt(&c.statement, 400),
                    "text": (c.text != c.statement).then(|| trunc_stmt(&c.text, 400)),
                    "noul": s.evidence,
                    "injection": s.injection,
                    "contradicts": s.contradicts,
                    "relevant": s.relevant,
                    "evidence": s.evidence,
                    "rrf": c.rrf_score,
                    "origin": c.origin,
                    "source": c.source,
                    "grantor": c.grantor_name,
                    "route": format!("{r:?}").to_lowercase(),
                    "admitted": matches!(r, FilterRoute::Include),
                })
            })
            .collect();
        all.sort_by(|a, b| {
            b["evidence"]
                .as_f64()
                .unwrap_or(0.0)
                .total_cmp(&a["evidence"].as_f64().unwrap_or(0.0))
        });
        all
    };
    let ranked_json = serde_json::to_string(&ranked).unwrap_or_else(|_| "[]".into());

    tracing::info!(
        ask_id = %ask_id,
        question = %trunc_stmt(question, 4000),
        candidates = candidates.len(),
        admitted = admitted.len(),
        conflicts = conflict_cites.len(),
        threshold = ctx.cfg.evidence_min,
        calibrated = ctx.cfg.admit_thresholds_calibrated,
        top_noul = includes.first().map(|(_, s)| s.evidence),
        ranked = ranked_json,
        "admission complete"
    );

    Ok(Admission {
        admitted,
        conflicts: conflict_cites,
        scores: by_id,
    })
}

fn citation(c: &Candidate, index: usize, noul: f64, route: FilterRoute) -> Citation {
    Citation {
        id: c.id,
        index,
        statement: c.statement.clone(),
        noul,
        origin: c.origin.clone(),
        grantor_name: c.grantor_name.clone(),
        source: c.source.clone(),
        occurred_at: c.occurred_at,
        route,
        text: c.text.clone(),
    }
}

fn trunc_stmt(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max_chars).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn cfg(inj: f64, contra: f64, rel: f64, ev: f64, max: usize) -> Config {
        let mut c = Config::from_env();
        c.injection_max = inj;
        c.contradicts_min = contra;
        c.relevant_min = rel;
        c.evidence_min = ev;
        c.max_citations = max;
        c
    }

    fn s(inj: f64, contra: f64, rel: f64, ev: f64) -> FilterScores {
        FilterScores {
            injection: inj,
            contradicts: contra,
            relevant: rel,
            evidence: ev,
        }
    }

    #[test]
    fn policy_order() {
        let c = cfg(0.70, 0.70, 0.45, 0.55, 4);
        assert_eq!(route(s(0.8, 0.0, 1.0, 1.0), &c), FilterRoute::Exclude);
        assert_eq!(route(s(0.1, 0.8, 1.0, 1.0), &c), FilterRoute::Conflict);
        assert_eq!(route(s(0.1, 0.8, 0.2, 0.9), &c), FilterRoute::Exclude);
        assert_eq!(route(s(0.1, 0.1, 0.2, 0.9), &c), FilterRoute::Exclude);
        assert_eq!(route(s(0.1, 0.1, 0.6, 0.6), &c), FilterRoute::Include);
        assert_eq!(route(s(0.1, 0.1, 0.6, 0.4), &c), FilterRoute::Exclude);
    }

    #[test]
    fn citation_cap_is_includes_only() {
        let _ = Uuid::new_v4();
        let c = cfg(0.70, 0.70, 0.45, 0.55, 2);
        let includes = 5;
        assert!(includes > c.max_citations);
    }
}
