//! Slack mrkdwn for answers. Ingested text is escaped so it cannot inject links or mentions.

use crate::{
    domain::{Cited, Provenance},
    pipeline::report::{Answer, Citation, Verdict},
};

/// The fixed decline message, shown whatever the decline reason.
pub const DECLINED: &str = "Can't answer with certainty.";

/// The answer in its category's format, with numbered sources and the run id last.
pub fn answer(answer: &Answer, run_id: &str) -> String {
    let markers = Markers(&answer.citations);
    let mut blocks = Vec::new();
    match &answer.verdict {
        Some(Verdict::Boolean { yes, confidence }) => {
            let verdict = if *yes { "Yes" } else { "No" };
            blocks.push(format!("*{verdict}* (confidence {:.2})", confidence.get()));
        }
        Some(Verdict::Choice { selected, rejected }) => {
            blocks.push(format!(
                "*{}*{}",
                text(&selected.value),
                markers.of(selected)
            ));
            for r in rejected {
                blocks.push(format!(
                    "• Not {}{}: {}",
                    text(&r.option.value),
                    markers.of(&r.option),
                    text(&r.reason)
                ));
            }
        }
        Some(Verdict::Ranking { criterion, items }) => {
            blocks.push(format!("Ranked by {}:", text(criterion)));
            for (i, item) in items.iter().enumerate() {
                blocks.push(format!(
                    "{}. {}{}",
                    i + 1,
                    text(&item.value),
                    markers.of(item)
                ));
            }
        }
        None => {}
    }
    if !answer.why.is_empty() {
        let claims: Vec<_> = answer
            .why
            .iter()
            .map(|claim| format!("{}{}", text(&claim.value), markers.of(claim)))
            .collect();
        blocks.push(claims.join(" "));
    }
    if !answer.citations.is_empty() {
        blocks.push(format!("*Sources*\n{}", sources(&answer.citations)));
    }
    blocks.push(format!("_Run {run_id}_"));
    blocks.join("\n\n")
}

/// [`DECLINED`], the closest sources when there are any, and the run id.
pub fn declined(sources_ranked: &[Citation], run_id: &str) -> String {
    let mut out = if sources_ranked.is_empty() {
        DECLINED.to_owned()
    } else {
        format!(
            "{DECLINED}\n\nThe closest sources, best first:\n\n{}",
            sources(sources_ranked)
        )
    };
    out.push_str(&format!("\n\n_Run {run_id}_"));
    out
}

struct Markers<'a>(&'a [Citation]);

impl Markers<'_> {
    fn of<T>(&self, cited: &Cited<T>) -> String {
        let mut numbers = Vec::new();
        let mut from_prompt = false;
        for source in cited.sources.as_slice() {
            match source {
                Provenance::Chunk(id) => {
                    if let Some(i) = self.0.iter().position(|c| &c.chunk.id == id)
                        && !numbers.contains(&(i + 1))
                    {
                        numbers.push(i + 1);
                    }
                }
                Provenance::UserPrompt(_) => from_prompt = true,
            }
        }
        let mut out: String = numbers.iter().map(|n| format!(" [{n}]")).collect();
        if from_prompt {
            out.push_str(" (from your question)");
        }
        out
    }
}

fn sources(citations: &[Citation]) -> String {
    citations
        .iter()
        .enumerate()
        .map(|(i, citation)| {
            let marker = format!("{}. ", i + 1);
            let title = text(&citation.title);
            let title = if title.is_empty() {
                "Untitled source".to_owned()
            } else {
                title
            };
            let mut item = format!("{marker}{}", link(&title, &citation.url));
            let excerpt = text(&citation.excerpt);
            if !excerpt.is_empty() {
                item.push_str(&format!("\n>{excerpt}"));
            }
            item
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn link(title: &str, url: &str) -> String {
    let url = url.trim();
    let scheme = url
        .split_once("://")
        .map(|(scheme, _)| scheme.to_ascii_lowercase());
    if matches!(scheme.as_deref(), Some("http" | "https")) {
        format!("<{url}|{title}>")
    } else {
        title.to_owned()
    }
}

fn text(raw: &str) -> String {
    let mut out = String::new();
    for (i, word) in raw.split_whitespace().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        for ch in word.chars() {
            match ch {
                '&' => out.push_str("&amp;"),
                '<' => out.push_str("&lt;"),
                '>' => out.push_str("&gt;"),
                ch => out.push(ch),
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Probability;

    #[test]
    fn ingested_text_cannot_inject_mrkdwn() {
        assert_eq!(
            text("<@U123|evil> & <https://evil.test|click>"),
            "&lt;@U123|evil&gt; &amp; &lt;https://evil.test|click&gt;"
        );
    }

    #[test]
    fn boolean_includes_run_id() {
        let answer = Answer {
            verdict: Some(Verdict::Boolean {
                yes: true,
                confidence: Probability::new(0.9).unwrap(),
            }),
            why: vec![],
            citations: vec![],
        };
        let rendered = super::answer(&answer, "019");
        assert!(rendered.contains("*Yes*"));
        assert!(rendered.contains("_Run 019_"));
    }
}
