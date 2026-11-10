//! Markdown for `choices[0].message.content`, in the per-category format of `docs/design.md` section 4.
//!
//! Every string from a run (claims, options, titles, excerpts) derives from ingested text, so it is escaped
//! and flattened to one line: it cannot add links, images, HTML, or block structure to the answer.

use crate::{
    domain::{Cited, Provenance},
    pipeline::report::{Answer, Citation, Verdict},
};

/// The fixed decline message, shown whatever the decline reason.
pub const DECLINED: &str = "Can't answer with certainty.";

/// The answer in its category's format: verdict first, then the why with `[n]` markers, then the numbered sources.
pub fn answer(answer: &Answer) -> String {
    let markers = Markers(&answer.citations);
    let mut blocks = Vec::new();
    match &answer.verdict {
        Some(Verdict::Boolean { yes, confidence }) => {
            let verdict = if *yes { "Yes" } else { "No" };
            blocks.push(format!(
                "**{verdict}** (confidence {:.2})",
                confidence.get()
            ));
        }
        Some(Verdict::Choice { selected, rejected }) => {
            blocks.push(format!(
                "**{}**{}",
                text(&selected.value),
                markers.of(selected)
            ));
            if !rejected.is_empty() {
                let lines: Vec<_> = rejected
                    .iter()
                    .map(|r| {
                        format!(
                            "- Not {}{}: {}",
                            text(&r.option.value),
                            markers.of(&r.option),
                            text(&r.reason)
                        )
                    })
                    .collect();
                blocks.push(lines.join("\n"));
            }
        }
        Some(Verdict::Ranking { criterion, items }) => {
            blocks.push(format!("Ranked by {}:", text(criterion)));
            let lines: Vec<_> = items
                .iter()
                .enumerate()
                .map(|(i, item)| format!("{}. {}{}", i + 1, text(&item.value), markers.of(item)))
                .collect();
            blocks.push(lines.join("\n"));
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
        blocks.push(format!("**Sources**\n\n{}", sources(&answer.citations)));
    }
    blocks.join("\n\n")
}

/// [`DECLINED`], followed by the closest sources when there are any.
pub fn declined(sources_ranked: &[Citation]) -> String {
    if sources_ranked.is_empty() {
        return DECLINED.into();
    }
    format!(
        "{DECLINED}\n\nThe closest sources, best first:\n\n{}",
        sources(sources_ranked)
    )
}

/// Maps provenance to the numbered citations of one answer.
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
    let items: Vec<_> = citations
        .iter()
        .enumerate()
        .map(|(i, citation)| {
            let marker = format!("{}. ", i + 1);
            let mut item = format!("{marker}{}", link(&citation.title, &citation.url));
            let excerpt = text(&citation.excerpt);
            if !excerpt.is_empty() {
                let indent = " ".repeat(marker.len());
                item.push_str(&format!("\n{indent}> {excerpt}"));
            }
            item
        })
        .collect();
    items.join("\n")
}

fn link(title: &str, url: &str) -> String {
    let title = match text(title) {
        title if title.is_empty() => "Untitled source".to_owned(),
        title => title,
    };
    let url = url.trim();
    let scheme = url
        .split_once("://")
        .map(|(scheme, _)| scheme.to_ascii_lowercase());
    if !matches!(scheme.as_deref(), Some("http" | "https")) {
        return title;
    }
    let url: String = url
        .chars()
        .map(|ch| match ch {
            ' ' | '<' | '>' | '\\' => format!("%{:02X}", ch as u32),
            ch if ch.is_control() => format!("%{:02X}", ch as u32),
            ch => ch.to_string(),
        })
        .collect();
    format!("[{title}](<{url}>)")
}

/// Escapes inline markdown and collapses whitespace, so the text renders literally on one line.
fn text(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for (i, word) in raw.split_whitespace().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        for ch in word.chars() {
            if matches!(
                ch,
                '\\' | '`' | '*' | '_' | '[' | ']' | '<' | '>' | '!' | '|' | '~' | '#' | '&'
            ) {
                out.push('\\');
            }
            out.push(ch);
        }
    }
    // Text can start a list item or paragraph, where `- `, `+ `, and `1. ` open a list.
    if out.starts_with(['-', '+']) {
        out.insert(0, '\\');
    }
    let digits = out.chars().take_while(char::is_ascii_digit).count();
    if digits > 0 && out[digits..].starts_with(['.', ')']) {
        out.insert(digits, '\\');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        domain::Probability,
        pipeline::report::Rejected,
        server::fixtures::{citation, cited, from_prompt},
    };

    #[test]
    fn boolean() {
        let answer = Answer {
            verdict: Some(Verdict::Boolean {
                yes: true,
                confidence: Probability::new(0.92).unwrap(),
            }),
            why: vec![
                cited("AGR_USERS assigns roles to users.", &[1]),
                cited("Assignments are time-bound.", &[2, 1]),
            ],
            citations: vec![citation(1), citation(2)],
        };
        assert_eq!(
            super::answer(&answer),
            "**Yes** (confidence 0.92)

AGR\\_USERS assigns roles to users. [1] Assignments are time-bound. [2] [1]

**Sources**

1. [Page 1](<https://wiki.example.test/1>)
   > Excerpt 1.
2. [Page 2](<https://wiki.example.test/2>)
   > Excerpt 2."
        );
    }

    #[test]
    fn choice() {
        let answer = Answer {
            verdict: Some(Verdict::Choice {
                selected: from_prompt("PFCG"),
                rejected: vec![Rejected {
                    option: from_prompt("SU01"),
                    reason: "it edits users, not roles".into(),
                }],
            }),
            why: vec![cited("Roles are maintained in PFCG.", &[1])],
            citations: vec![citation(1)],
        };
        assert_eq!(
            super::answer(&answer),
            "**PFCG** (from your question)

- Not SU01 (from your question): it edits users, not roles

Roles are maintained in PFCG. [1]

**Sources**

1. [Page 1](<https://wiki.example.test/1>)
   > Excerpt 1."
        );
    }

    #[test]
    fn ranking() {
        let answer = Answer {
            verdict: Some(Verdict::Ranking {
                criterion: "risk".into(),
                items: vec![cited("SAP_ALL", &[2]), cited("S_TCODE", &[1])],
            }),
            why: vec![cited("SAP_ALL grants everything.", &[2])],
            citations: vec![citation(1), citation(2)],
        };
        assert_eq!(
            super::answer(&answer),
            "Ranked by risk:

1. SAP\\_ALL [2]
2. S\\_TCODE [1]

SAP\\_ALL grants everything. [2]

**Sources**

1. [Page 1](<https://wiki.example.test/1>)
   > Excerpt 1.
2. [Page 2](<https://wiki.example.test/2>)
   > Excerpt 2."
        );
    }

    #[test]
    fn open() {
        let answer = Answer {
            verdict: None,
            why: vec![
                cited("ARM syncs roles nightly.", &[1]),
                cited("Failures land in the error queue.", &[1]),
            ],
            citations: vec![citation(1)],
        };
        assert_eq!(
            super::answer(&answer),
            "ARM syncs roles nightly. [1] Failures land in the error queue. [1]

**Sources**

1. [Page 1](<https://wiki.example.test/1>)
   > Excerpt 1."
        );
    }

    #[test]
    fn declined_lists_ranked_sources() {
        assert_eq!(
            super::declined(&[citation(2), citation(1)]),
            "Can't answer with certainty.

The closest sources, best first:

1. [Page 2](<https://wiki.example.test/2>)
   > Excerpt 2.
2. [Page 1](<https://wiki.example.test/1>)
   > Excerpt 1."
        );
        assert_eq!(super::declined(&[]), DECLINED);
    }

    #[test]
    fn ingested_text_cannot_add_markup() {
        assert_eq!(
            text("![x](https://evil.test/?q=1)\n# <b>hi</b>"),
            "\\!\\[x\\](https://evil.test/?q=1) \\# \\<b\\>hi\\</b\\>"
        );
        assert_eq!(text("- item"), "\\- item");
        assert_eq!(text("2024. A year"), "2024\\. A year");
        assert_eq!(link("Page", "javascript:alert(1)"), "Page");
        assert_eq!(
            link("[Page]", "https://x.test/a b>\\"),
            "[\\[Page\\]](<https://x.test/a%20b%3E%5C>)"
        );
    }
}
