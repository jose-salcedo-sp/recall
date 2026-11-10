//! Question expansion through the corpus's own vocabulary registers (DaaS `canon/GLOSSARY.md` and
//! `customer/LEXICON.md`): a question that uses one side of a register row also searches with the row's other words.
//!
//! The registers are ordinary ingested chunks, parsed on each retrieval, so an edit takes effect at the next ingest
//! without a separate table. Only the words of pipe-table rows are kept: the map holds terms, never instructions.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::domain::ChunkRef;

/// Words too common to add to a search. The full-text side drops English stop words anyway; listing them here keeps
/// them from using up the cap.
const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from", "if", "in", "is", "it",
    "its", "just", "no", "not", "of", "on", "or", "re", "s", "say", "t", "that", "the", "then",
    "this", "to", "use", "we", "with",
];

/// The rows of every register table, in register order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Glossary {
    entries: Vec<Entry>,
}

#[derive(Clone, Debug, PartialEq)]
struct Entry {
    source: ChunkRef,
    keys: Vec<Key>,
    /// Lowercased words of every cell, first cell first, without stop words or duplicates.
    terms: Vec<String>,
}

/// How a question selects an entry.
#[derive(Clone, Debug, PartialEq)]
enum Key {
    /// Two or more lowercased words, matched as consecutive question words.
    Phrase(Vec<String>),
    /// A lowercased acronym, matched only by a question word that [`is_acronym`] (`SoD`, `NEW`), so the ordinary
    /// word `new` does not select it.
    Acronym(String),
}

/// The register entries one question matched, in register order, and the terms each added.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Expansion(Vec<GlossaryMatch>);

/// One register entry a question matched.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GlossaryMatch {
    /// The question words that selected the entry, lowercased.
    pub matched: String,
    /// The register chunk the entry was read from.
    pub source: ChunkRef,
    /// The words the entry added to the search; empty when the cap was already reached.
    pub terms: Vec<String>,
}

impl Expansion {
    /// The matched entries, in register order.
    pub fn matches(&self) -> &[GlossaryMatch] {
        &self.0
    }

    /// Every added term, in order.
    pub fn terms(&self) -> impl Iterator<Item = &str> {
        self.0
            .iter()
            .flat_map(|m| m.terms.iter().map(String::as_str))
    }

    /// The question followed by the added terms.
    pub fn apply(&self, question: &str) -> String {
        std::iter::once(question)
            .chain(self.terms())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Whether the question matched no entry.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Glossary {
    /// Reads the pipe tables of `chunks` (in the order given): each body row below a table's header is one entry,
    /// selected by its first cell and expanding to the words of all its cells. Text in backticks is code, so it
    /// neither selects an entry nor is added; bracketed provenance tags such as `[SME 2026-10-09]` are dropped.
    pub fn parse<'a>(chunks: impl IntoIterator<Item = (ChunkRef, &'a str)>) -> Self {
        let entries = chunks
            .into_iter()
            .flat_map(|(source, text)| {
                table_rows(text)
                    .into_iter()
                    .filter_map(move |cells| Entry::new(source.clone(), &cells))
            })
            .collect();
        Self { entries }
    }

    /// Whether no table row was found.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entries `question` matches and the words they add, at most `max_terms` in total. Words already in the
    /// question are not added. The same question and glossary always give the same expansion.
    pub fn expand(&self, question: &str, max_terms: usize) -> Expansion {
        let question_words = words(question);
        let acronyms: HashSet<String> = question
            .split(|c: char| !c.is_alphanumeric())
            .filter(|word| is_acronym(word))
            .map(str::to_lowercase)
            .collect();
        let mut seen: HashSet<&str> = question_words.iter().map(String::as_str).collect();
        let mut budget = max_terms;
        let mut matches = Vec::new();
        for entry in &self.entries {
            let Some(matched) = entry.keys.iter().find_map(|key| match key {
                Key::Phrase(phrase) => question_words
                    .windows(phrase.len())
                    .any(|window| window == phrase.as_slice())
                    .then(|| phrase.join(" ")),
                Key::Acronym(acronym) => acronyms.contains(acronym).then(|| acronym.clone()),
            }) else {
                continue;
            };
            let mut terms = Vec::new();
            for term in &entry.terms {
                if budget == 0 {
                    break;
                }
                if seen.insert(term) {
                    terms.push(term.clone());
                    budget -= 1;
                }
            }
            matches.push(GlossaryMatch {
                matched,
                source: entry.source.clone(),
                terms,
            });
        }
        Expansion(matches)
    }
}

impl Entry {
    fn new(source: ChunkRef, cells: &[&str]) -> Option<Self> {
        let first = without_code(cells.first()?);
        let quoted: Vec<&str> = first.split('"').skip(1).step_by(2).collect();
        let variants: Vec<String> = if quoted.is_empty() {
            without_parentheses(&first)
                .split(['/', ','])
                .map(str::to_owned)
                .collect()
        } else {
            quoted.into_iter().map(str::to_owned).collect()
        };
        // A question that names the whole term is already searched by it, and the corpus uses that name; only a part,
        // an alternative or an abbreviation of the term is bridged to the rest of the row.
        let name = words(&first);
        let mut keys: Vec<Key> = variants
            .iter()
            .map(|variant| words(variant))
            .filter(|phrase| phrase.len() >= 2 && *phrase != name)
            .map(Key::Phrase)
            .collect();
        keys.extend(
            acronyms(&first)
                .filter(|acronym| name != [acronym.as_str()])
                .map(Key::Acronym),
        );
        if keys.is_empty() {
            return None;
        }
        let mut seen = HashSet::new();
        let terms = cells
            .iter()
            .flat_map(|cell| words(&without_code(cell)))
            .filter(|word| !STOP_WORDS.contains(&word.as_str()) && seen.insert(word.clone()))
            .collect();
        Some(Self {
            source,
            keys,
            terms,
        })
    }
}

/// Acronyms a cell names: words that [`is_acronym`] (`MSRB`, `NEW`), and the initials of a compound of three
/// or more hyphenated words (`Segregation-of-duties` gives `sod`).
fn acronyms(cell: &str) -> impl Iterator<Item = String> + '_ {
    cell.split(|c: char| !(c.is_alphanumeric() || c == '-'))
        .flat_map(|token| {
            let parts: Vec<&str> = token.split('-').filter(|part| !part.is_empty()).collect();
            let initials = (parts.len() >= 3).then(|| {
                parts
                    .iter()
                    .filter_map(|part| part.chars().next())
                    .collect::<String>()
            });
            parts
                .into_iter()
                .filter(|part| is_acronym(part))
                .map(str::to_owned)
                .chain(initials)
                .collect::<Vec<_>>()
        })
        .map(|acronym| acronym.to_lowercase())
}

/// Two or more capitals and no more lower-case letters than capitals: `SoD` and `NEW`, not `ErpMaestro`.
fn is_acronym(word: &str) -> bool {
    let upper = word.chars().filter(|c| c.is_uppercase()).count();
    upper >= 2 && upper >= word.chars().filter(|c| c.is_lowercase()).count()
}

/// Body rows of the pipe tables in `text`, as trimmed cells. A table's rows above its `|---|` line are its header.
fn table_rows(text: &str) -> Vec<Vec<&str>> {
    let mut rows = Vec::new();
    let mut in_body = false;
    for line in text.lines().map(str::trim) {
        let Some(inner) = line.strip_prefix('|') else {
            in_body = false;
            continue;
        };
        let cells: Vec<&str> = inner
            .strip_suffix('|')
            .unwrap_or(inner)
            .split('|')
            .map(str::trim)
            .collect();
        let separator = cells.iter().all(|cell| {
            let dashes = cell.trim_matches(':');
            !dashes.is_empty() && dashes.chars().all(|c| c == '-')
        });
        if separator {
            in_body = true;
        } else if in_body {
            rows.push(cells);
        }
    }
    rows
}

/// `text` without backtick spans and bracketed tags.
fn without_code(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut closing = None;
    for c in text.chars() {
        match (closing, c) {
            (None, '`') => closing = Some('`'),
            (None, '[') => closing = Some(']'),
            (Some(end), c) if c == end => closing = None,
            (Some(_), _) => {}
            (None, c) => out.push(c),
        }
    }
    out
}

fn without_parentheses(text: &str) -> String {
    let mut depth = 0usize;
    text.chars()
        .filter(|&c| {
            match c {
                '(' => depth += 1,
                ')' => depth = depth.saturating_sub(1),
                _ => return depth == 0,
            }
            false
        })
        .collect()
}

/// Lowercased alphanumeric words. Words with a digit or underscore are identifiers, which the full-text side matches
/// exactly from the question itself, so they are left out.
fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty() && !word.chars().any(|c| c.is_ascii_digit()))
        .map(str::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{ChunkId, ChunkVersion};

    fn source(name: &str) -> ChunkRef {
        ChunkRef {
            id: ChunkId(format!("markdown:{name}")),
            version: ChunkVersion("v1".into()),
        }
    }

    const LEXICON: &str = "Intro with | a stray pipe.

| Internal | Say instead |
|---|---|
| Legacy / NEW generation | \"the current screens\" and \"the earlier screens\" `[SME 2026-10-05]` |
| `BUG-nn` | \"an issue we're tracking\" |
| \"Known issue\" with no qualifier | \"An issue we're aware of\" |";

    const GLOSSARY: &str = "| Term | Meaning |
|---|---|
| **Segregation-of-duties risk** | Two conflicting business functions |
| **Risk** | A5 `Risk Code` |
| **EAM** | Emergency Access Management |
| **ERPM / ErpMaestro** | The pre-SailPoint product name |";

    fn glossary() -> Glossary {
        Glossary::parse([
            (source("LEXICON.md"), LEXICON),
            (source("GLOSSARY.md"), GLOSSARY),
        ])
    }

    #[test]
    fn header_rows_code_cells_single_words_and_whole_terms_select_nothing() {
        let glossary = glossary();
        assert_eq!(glossary.entries.len(), 4);
        for question in [
            "What does Internal mean?",
            "BUG-12 again",
            "Is this a risk?",
            "a new screen",
            "Who reviews EAM requests?",
            "Check ErpMaestro.WebUI",
        ] {
            assert_eq!(
                glossary.expand(question, 100),
                Expansion::default(),
                "{question}"
            );
        }
    }

    #[test]
    fn an_acronym_needs_capitals_in_the_question() {
        let expansion = glossary().expand("Must a user hold both sides of an SoD risk?", 100);
        assert_eq!(expansion.matches().len(), 1);
        assert_eq!(expansion.matches()[0].matched, "sod");
        assert_eq!(expansion.matches()[0].source, source("GLOSSARY.md"));
        assert_eq!(
            expansion.terms().collect::<Vec<_>>(),
            [
                "segregation",
                "duties",
                "two",
                "conflicting",
                "business",
                "functions"
            ]
        );

        let expansion = glossary().expand("Only on the NEW screen?", 100);
        assert_eq!(
            expansion.terms().collect::<Vec<_>>(),
            ["legacy", "generation", "current", "screens", "earlier"]
        );
        assert_eq!(
            expansion.apply("Only on the NEW screen?"),
            "Only on the NEW screen? legacy generation current screens earlier"
        );
    }

    #[test]
    fn a_quoted_phrase_selects_its_row() {
        let expansion = glossary().expand("Can I call it a known issue?", 100);
        assert_eq!(expansion.matches()[0].matched, "known issue");
        assert_eq!(
            expansion.terms().collect::<Vec<_>>(),
            ["qualifier", "aware"]
        );
    }

    #[test]
    fn an_alternative_name_selects_its_row() {
        let expansion = glossary().expand("Where does ERPM survive?", 100);
        assert_eq!(expansion.matches()[0].matched, "erpm");
        assert_eq!(
            expansion.terms().collect::<Vec<_>>(),
            ["erpmaestro", "pre", "sailpoint", "product", "name"]
        );
    }

    #[test]
    fn expansion_is_capped_in_register_order_and_deterministic() {
        let question = "SoD risk on the NEW screen";
        let expansion = glossary().expand(question, 4);
        assert_eq!(expansion.terms().count(), 4);
        let matched: Vec<_> = expansion
            .matches()
            .iter()
            .map(|m| m.matched.as_str())
            .collect();
        assert_eq!(matched, ["new", "sod"]);
        assert!(expansion.matches()[1].terms.is_empty());
        assert_eq!(expansion, glossary().expand(question, 4));
        assert_eq!(glossary().expand(question, 0).terms().count(), 0);
    }
}
