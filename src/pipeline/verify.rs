use std::collections::HashSet;
use std::sync::Arc;

use crate::ctx::Ctx;
use crate::error::Result;
use crate::types::{AdmittedCitations, Citation, ClaimVerdict, Stage, EMPTY_ADMIT_TEXT};

/// Split on period, question mark, or newline. No LLM splitter.
pub fn split_claims(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    for ch in text.chars() {
        buf.push(ch);
        if matches!(ch, '.' | '?' | '\n') {
            let t = buf.trim().to_string();
            if !t.is_empty() {
                out.push(t);
            }
            buf.clear();
        }
    }
    let t = buf.trim().to_string();
    if !t.is_empty() {
        out.push(t);
    }
    out
}

fn memory_cites(claim: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let bytes = claim.as_bytes();
    let mut i = 0;
    while i + 8 < bytes.len() {
        if claim[i..].starts_with("[memory_") {
            let rest = &claim[i + 8..];
            if let Some(end) = rest.find(']') {
                if let Ok(n) = rest[..end].parse::<usize>() {
                    out.push(n);
                }
                i += 8 + end + 1;
                continue;
            }
        }
        i += 1;
    }
    out
}

fn quotes(claim: &str) -> Vec<String> {
    let mut out = Vec::new();
    let chars: Vec<char> = claim.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if matches!(chars[i], '"' | '“' | '\'') {
            let open = chars[i];
            let close = match open {
                '“' => '”',
                _ => open,
            };
            if let Some(rel) = chars[i + 1..].iter().position(|&c| c == close) {
                let q: String = chars[i + 1..i + 1 + rel].iter().collect();
                if !q.trim().is_empty() {
                    out.push(q);
                }
                i += rel + 2;
                continue;
            }
        }
        i += 1;
    }
    out
}

fn content_tokens(s: &str) -> HashSet<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.chars().count() > 2)
        .map(|t| t.to_lowercase())
        .collect()
}

fn shared_tokens(claim: &HashSet<String>, section: &str) -> usize {
    content_tokens(section).intersection(claim).count()
}

/// Admitted section a citation-less claim is actually talking about.
///
/// The 1.5B generator paraphrases an admitted memory and omits `[memory_N]`.
/// Two or more shared content words is enough to send that section to relate;
/// a greeting or a persona sentence shares nothing and stays unsupported.
pub fn attribute_uncited<'a>(
    claim: &str,
    cites: &'a AdmittedCitations,
) -> Option<(usize, &'a str)> {
    let toks = content_tokens(claim);
    if toks.len() < 2 {
        return None;
    }
    let mut best: Option<(usize, &str, usize, f64)> = None;
    for c in cites.all() {
        for section in [c.text.as_str(), c.statement.as_str()] {
            let section = section.trim();
            if section.is_empty() {
                continue;
            }
            let shared = shared_tokens(&toks, section);
            if shared < 2 {
                continue;
            }
            let score = shared as f64 / toks.len() as f64;
            let replace = match best {
                None => true,
                Some((_, _, b_shared, b_score)) => {
                    score > b_score + 1e-9 || ((score - b_score).abs() <= 1e-9 && shared > b_shared)
                }
            };
            if replace {
                best = Some((c.index, section, shared, score));
            }
        }
    }
    best.map(|(index, section, _, _)| (index, section))
}

fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_whitespace() && !matches!(c, '"' | '\'' | '“' | '”'))
        .flat_map(|c| c.to_lowercase())
        .collect()
}

fn in_text(span: &str, text: &str) -> bool {
    let n = normalize(span);
    !n.is_empty() && normalize(text).contains(&n)
}

/// Local checks only. `fabricated` means skip System One.
pub fn local_verdict(claim: &str, cites_set: &AdmittedCitations) -> Option<ClaimVerdict> {
    let ms = memory_cites(claim);
    if ms.is_empty() {
        return Some(ClaimVerdict {
            claim: claim.to_string(),
            verdict: "unsupported".into(),
            memory_index: None,
        });
    }
    let qs = quotes(claim);
    if qs.is_empty() {
        return Some(ClaimVerdict {
            claim: claim.to_string(),
            verdict: "fabricated".into(),
            memory_index: ms.first().copied(),
        });
    }
    let ok = qs.iter().all(|q| {
        ms.iter().any(|i| {
            cites_set
                .by_index(*i)
                .map(|c| in_text(q, &c.text))
                .unwrap_or(false)
        })
    });
    if !ok {
        return Some(ClaimVerdict {
            claim: claim.to_string(),
            verdict: "fabricated".into(),
            memory_index: ms.first().copied(),
        });
    }
    None
}

pub fn strip_failed(text: &str, verdicts: &[ClaimVerdict]) -> String {
    let fail: HashSet<&str> = verdicts
        .iter()
        .filter(|v| v.verdict != "supports")
        .map(|v| v.claim.as_str())
        .collect();
    split_claims(text)
        .into_iter()
        .filter(|c| !fail.contains(c.as_str()))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Memory indices referenced by `[memory_N]` markers in published text.
pub fn cited_indices(text: &str) -> HashSet<usize> {
    split_claims(text)
        .iter()
        .flat_map(|c| memory_cites(c))
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishOutcome {
    pub text: String,
    pub empty: bool,
}

/// Strip failed claims and drop citations not referenced by surviving text.
pub fn apply_publish(
    raw_text: &str,
    verdicts: &[ClaimVerdict],
    admitted: &mut Vec<Citation>,
    conflicts: &mut Vec<Citation>,
) -> PublishOutcome {
    let stripped = strip_failed(raw_text, verdicts);
    if stripped.trim().is_empty() {
        admitted.clear();
        conflicts.clear();
        return PublishOutcome {
            text: EMPTY_ADMIT_TEXT.to_string(),
            empty: true,
        };
    }
    let mut cited = cited_indices(&stripped);
    for v in verdicts {
        if v.verdict == "supports" {
            if let Some(i) = v.memory_index {
                if stripped.contains(v.claim.as_str()) {
                    cited.insert(i);
                }
            }
        }
    }
    admitted.retain(|c| cited.contains(&c.index));
    conflicts.retain(|c| cited.contains(&c.index));
    PublishOutcome {
        text: stripped,
        empty: false,
    }
}

pub async fn run(
    ctx: &Arc<Ctx>,
    text: &str,
    cites: &AdmittedCitations,
) -> Result<Vec<ClaimVerdict>> {
    let mut out = Vec::new();
    for claim in split_claims(text) {
        let (idx, section) = if let Some(v) = local_verdict(&claim, cites) {
            if v.verdict == "unsupported" && v.memory_index.is_none() {
                match attribute_uncited(&claim, cites) {
                    Some((idx, section)) => (idx, section.to_string()),
                    None => {
                        out.push(v);
                        continue;
                    }
                }
            } else {
                out.push(v);
                continue;
            }
        } else {
            let idx = memory_cites(claim.as_str())[0];
            let section = cites
                .by_index(idx)
                .map(|c| c.text.clone())
                .unwrap_or_default();
            (idx, section)
        };
        let rel = ctx
            .with_retry(Stage::Verify, ctx.cfg.admit_timeout, || {
                ctx.system_one.relate(&claim, &section)
            })
            .await?;
        let verdict = if rel.confidence < ctx.cfg.cite_confidence_min {
            "unsupported".into()
        } else {
            rel.choice
        };
        out.push(ClaimVerdict {
            claim,
            verdict,
            memory_index: Some(idx),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Citation, FilterRoute, EMPTY_ADMIT_TEXT};
    use uuid::Uuid;

    fn citeset(text: &str) -> AdmittedCitations {
        AdmittedCitations::new(
            vec![Citation {
                id: Uuid::from_u128(1),
                index: 0,
                statement: text.into(),
                noul: 1.0,
                origin: "personal".into(),
                grantor_name: None,
                source: None,
                occurred_at: None,
                route: FilterRoute::Include,
                text: text.into(),
            }],
            vec![],
        )
        .unwrap()
    }

    #[test]
    fn missing_cite_is_unsupported() {
        let v = local_verdict("Ana was born in March.", &citeset("Ana March 14")).unwrap();
        assert_eq!(v.verdict, "unsupported");
    }

    #[test]
    fn missing_quote_is_fabricated() {
        let v = local_verdict("Ana was born in March [memory_0].", &citeset("March 14")).unwrap();
        assert_eq!(v.verdict, "fabricated");
    }

    #[test]
    fn quote_not_in_cited_memory_is_fabricated() {
        let c = citeset("Ana, my younger sister, was born on March 14.");
        let v = local_verdict("The badge PIN is \"3301\" [memory_0].", &c).unwrap();
        assert_eq!(v.verdict, "fabricated");
    }

    #[test]
    fn matching_quote_needs_choice() {
        let c = citeset("Ana, my younger sister, was born on March 14.");
        assert!(local_verdict("Ana was born on \"March 14\" [memory_0].", &c,).is_none());
    }

    #[test]
    fn strip_drops_failed_sentences() {
        let text = "Keep this [memory_0]. Drop this.";
        let vs = vec![
            ClaimVerdict {
                claim: "Keep this [memory_0].".into(),
                verdict: "supports".into(),
                memory_index: Some(0),
            },
            ClaimVerdict {
                claim: "Drop this.".into(),
                verdict: "unsupported".into(),
                memory_index: None,
            },
        ];
        assert_eq!(strip_failed(text, &vs), "Keep this [memory_0].");
    }

    fn cite(index: usize, text: &str) -> Citation {
        Citation {
            id: Uuid::from_u128(index as u128 + 1),
            index,
            statement: text.into(),
            noul: 1.0,
            origin: "personal".into(),
            grantor_name: None,
            source: None,
            occurred_at: None,
            route: FilterRoute::Include,
            text: text.into(),
        }
    }

    #[test]
    fn apply_publish_drops_uncited_citations() {
        let text = "Only zero [memory_0].";
        let vs = vec![ClaimVerdict {
            claim: "Only zero [memory_0].".into(),
            verdict: "supports".into(),
            memory_index: Some(0),
        }];
        let mut admitted = vec![cite(0, "a"), cite(1, "b")];
        let mut conflicts = vec![];
        let out = apply_publish(text, &vs, &mut admitted, &mut conflicts);
        assert_eq!(out.text, "Only zero [memory_0].");
        assert!(!out.empty);
        assert_eq!(admitted.len(), 1);
        assert_eq!(admitted[0].index, 0);
    }

    #[test]
    fn apply_publish_all_fail_becomes_empty_admit() {
        let text = "Bad [memory_0]. Also bad [memory_1].";
        let vs = vec![
            ClaimVerdict {
                claim: "Bad [memory_0].".into(),
                verdict: "unsupported".into(),
                memory_index: Some(0),
            },
            ClaimVerdict {
                claim: "Also bad [memory_1].".into(),
                verdict: "fabricated".into(),
                memory_index: Some(1),
            },
        ];
        let mut admitted = vec![cite(0, "a"), cite(1, "b")];
        let mut conflicts = vec![];
        let out = apply_publish(text, &vs, &mut admitted, &mut conflicts);
        assert_eq!(out.text, EMPTY_ADMIT_TEXT);
        assert!(out.empty);
        assert!(admitted.is_empty());
        assert!(conflicts.is_empty());
    }

    #[test]
    fn uncited_paraphrase_attributes_the_matching_memory() {
        let cites = AdmittedCitations::new(
            vec![
                cite(0, "Speaker 00 experienced heavy traffic near their house."),
                cite(1, "User is working on a Rust project using Cargo"),
            ],
            vec![],
        )
        .unwrap();
        let (idx, section) =
            attribute_uncited("You are working on a Rust project using Cargo.", &cites).unwrap();
        assert_eq!(idx, 1);
        assert!(section.contains("Cargo"));
    }

    #[test]
    fn persona_sentence_is_not_attributed() {
        let cites = citeset("Speaker 00 experienced heavy traffic near their house.");
        assert!(attribute_uncited("I am the memory of the user, Speaker 00.", &cites).is_none());
    }

    #[test]
    fn apply_publish_keeps_citation_from_memory_index() {
        let text = "You are working on a Rust project using Cargo.";
        let vs = vec![ClaimVerdict {
            claim: text.into(),
            verdict: "supports".into(),
            memory_index: Some(1),
        }];
        let mut admitted = vec![cite(0, "traffic"), cite(1, "cargo")];
        let mut conflicts = vec![];
        let out = apply_publish(text, &vs, &mut admitted, &mut conflicts);
        assert_eq!(out.text, text);
        assert!(!out.empty);
        assert_eq!(admitted.len(), 1);
        assert_eq!(admitted[0].index, 1);
    }
}
