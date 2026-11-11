//! Ingest pre-audited markdown files from a directory: one document per file, one chunk per heading section.

use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    time::SystemTime,
};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use reqwest::Url;

use crate::{
    access::Acl,
    domain::{ChunkDraft, Document, Labels, Sensitivity, SourceName, Standing},
};

use super::Source;

const INTRO: &str = "intro";

/// The [`Source::name`] of the markdown connector, and the prefix of its document ids.
pub static NAME: SourceName = SourceName::from_static("markdown");

/// Where the markdown corpus is and how its paths split into layers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MarkdownConfig {
    /// Root directory of the markdown corpus.
    pub docs: PathBuf,
    /// Optional base URL for citations (e.g. a GitHub blob URL ending in `/`).
    pub base_url: Option<String>,
    /// Paths under the docs root holding settled, authoritative answers. Empty means the whole root.
    pub authoritative: Vec<PathBuf>,
    /// Paths under the docs root ingested as supporting evidence, not authoritative. The longest listed path
    /// containing a file decides its layer; files under no listed path are skipped.
    pub evidence: Vec<PathBuf>,
    /// Paths under the docs root to leave out, even inside an authoritative or evidence path.
    pub skip: Vec<PathBuf>,
}

/// The markdown [`Source`]: walks the docs root for `*.md` files, skipping dot-directories, and turns each file in a
/// layer into a [`Document`].
#[derive(Clone)]
pub struct Connector {
    root: PathBuf,
    base_url: Option<String>,
    /// `(path relative to root, standing)`; `None` skips the path.
    layers: Vec<(PathBuf, Option<Standing>)>,
}

impl Connector {
    /// The corpus `config` describes.
    ///
    /// # Errors
    ///
    /// When the docs root is not a directory, the base URL does not parse, or a layer path is absolute, contains
    /// `..`, does not exist under the root, or is listed twice.
    pub fn new(config: &MarkdownConfig) -> eros::Result<Self> {
        let root = config.docs.clone();
        eros::ensure!(
            root.is_dir(),
            "docs path {} is not a directory",
            root.display()
        );
        if let Some(base) = &config.base_url {
            Url::parse(base).map_err(|e| eros::error!("invalid docs base URL: {e}"))?;
        }
        let listed = |paths: &[PathBuf]| -> Vec<PathBuf> {
            paths
                .iter()
                .filter(|p| !p.as_os_str().is_empty())
                .cloned()
                .collect()
        };
        let mut authoritative = listed(&config.authoritative);
        if authoritative.is_empty() {
            authoritative.push(PathBuf::new());
        }
        let layers: Vec<(PathBuf, Option<Standing>)> = authoritative
            .into_iter()
            .map(|p| (p, Some(Standing::Authoritative)))
            .chain(
                listed(&config.evidence)
                    .into_iter()
                    .map(|p| (p, Some(Standing::Evidence))),
            )
            .chain(listed(&config.skip).into_iter().map(|p| (p, None)))
            .collect();
        for (path, _) in &layers {
            eros::ensure!(
                path.is_relative()
                    && path
                        .components()
                        .all(|c| matches!(c, std::path::Component::Normal(_))),
                "corpus path {} must be relative to the docs root, without `..`",
                path.display()
            );
            eros::ensure!(
                root.join(path).exists(),
                "corpus path {} does not exist under {}",
                path.display(),
                root.display()
            );
            eros::ensure!(
                layers.iter().filter(|(p, _)| p == path).count() == 1,
                "corpus path {} is listed more than once",
                path.display()
            );
        }
        Ok(Self {
            root,
            base_url: config.base_url.clone(),
            layers,
        })
    }

    /// `None` when the file is skipped or outside every listed path, so not part of the corpus.
    fn standing(&self, rel: &Path) -> Option<Standing> {
        self.layers
            .iter()
            .filter(|(path, _)| rel.starts_with(path))
            .max_by_key(|(path, _)| path.components().count())
            .and_then(|(_, standing)| *standing)
    }

    fn document(&self, path: &Path, standing: Standing) -> eros::Result<Document> {
        let rel = path
            .strip_prefix(&self.root)
            .map_err(|_| eros::error!("path outside docs root"))?;
        let external_id = rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        let text = fs::read_to_string(path)?;
        let title = document_title(&text, path);
        let updated_at = file_mtime(path)?;
        let file_url = self.file_url(&external_id);
        let chunks = chunks(&file_url, &text, standing)?;
        Ok(Document {
            source: NAME.clone(),
            external_id,
            title,
            url: file_url,
            updated_at,
            labels: Labels {
                acl: Acl::anyone(),
                sensitivity: Sensitivity::Internal,
            },
            chunks,
        })
    }

    fn file_url(&self, external_id: &str) -> String {
        match &self.base_url {
            Some(base) => format!("{}/{external_id}", base.trim_end_matches('/')),
            None => external_id.to_owned(),
        }
    }

    fn collect_md_files(&self) -> eros::Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        collect_md_recursive(&self.root, &mut files)?;
        files.sort();
        Ok(files)
    }
}

fn collect_md_recursive(dir: &Path, out: &mut Vec<PathBuf>) -> eros::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        if path.is_dir() {
            if name.to_string_lossy().starts_with('.') {
                continue;
            }
            collect_md_recursive(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "md") {
            out.push(path);
        }
    }
    Ok(())
}

fn file_mtime(path: &Path) -> eros::Result<DateTime<Utc>> {
    let meta = fs::metadata(path)?;
    let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    Ok(DateTime::<Utc>::from(modified))
}

/// First `# ` heading in the file, else the file stem.
fn document_title(content: &str, path: &Path) -> String {
    for line in content.lines() {
        if let Some(title) = atx_heading(line)
            && title.level == 1
        {
            return title.text;
        }
    }
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Untitled".into())
}

struct AtxTitle {
    level: usize,
    text: String,
}

fn atx_heading(line: &str) -> Option<AtxTitle> {
    let trimmed = line.trim();
    let hashes = trimmed.chars().take_while(|&c| c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let text = trimmed[hashes..].trim().to_owned();
    if text.is_empty() {
        return None;
    }
    Some(AtxTitle {
        level: hashes,
        text,
    })
}

#[derive(Debug, PartialEq)]
struct Section {
    path: Vec<String>,
    text: String,
}

/// One chunk per non-empty heading section; headings inside fenced code blocks are ignored. Front matter is
/// dropped. Text before the first heading becomes the `intro` chunk. Section URLs get a GitHub-style anchor only
/// when `file_url` is absolute.
///
/// # Errors
///
/// None today: every input splits into sections.
pub fn chunks(file_url: &str, content: &str, standing: Standing) -> eros::Result<Vec<ChunkDraft>> {
    let body = strip_front_matter(content);
    let sections = sections(body)?;
    let mut used = HashSet::from([INTRO.to_owned()]);
    let mut drafts = Vec::new();
    for section in sections {
        if section.text.trim().is_empty() {
            continue;
        }
        let (locator, heading, url) = match section.path.last() {
            None => (INTRO.to_owned(), None, file_url.to_owned()),
            Some(title) => {
                let fragment = github_anchor(title);
                let url = if file_url.contains("://") {
                    format!("{file_url}#{fragment}")
                } else {
                    file_url.to_owned()
                };
                (
                    unique(&slug(title), &mut used),
                    Some(section.path.join(" > ")),
                    url,
                )
            }
        };
        drafts.push(ChunkDraft {
            locator,
            heading,
            url,
            text: section.text.trim().to_owned(),
            standing,
        });
    }
    Ok(drafts)
}

fn strip_front_matter(content: &str) -> &str {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return content;
    }
    let after = trimmed.strip_prefix("---").unwrap_or(trimmed);
    if let Some(end) = after.find("\n---") {
        let rest = &after[end + 4..];
        rest.strip_prefix('\n').unwrap_or(rest)
    } else {
        content
    }
}

fn sections(content: &str) -> eros::Result<Vec<Section>> {
    let mut out = Vec::new();
    let mut path: Vec<String> = Vec::new();
    let mut current_text = String::new();
    let mut in_fence = false;
    let mut fence_marker: Option<String> = None;

    let mut flush = |path: &Vec<String>, text: &mut String| {
        out.push(Section {
            path: path.clone(),
            text: std::mem::take(text),
        });
    };

    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(marker) = fence_marker.as_ref() {
            if trimmed.starts_with(marker.as_str()) {
                in_fence = false;
                fence_marker = None;
            }
            current_text.push_str(line);
            current_text.push('\n');
            continue;
        }
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            let marker = if trimmed.starts_with("```") {
                "```"
            } else {
                "~~~"
            };
            in_fence = true;
            fence_marker = Some(marker.to_owned());
            current_text.push_str(line);
            current_text.push('\n');
            continue;
        }
        if !in_fence && let Some(h) = atx_heading(line) {
            flush(&path, &mut current_text);
            while path.len() >= h.level {
                path.pop();
            }
            path.push(h.text);
            continue;
        }
        current_text.push_str(line);
        current_text.push('\n');
    }
    flush(&path, &mut current_text);
    Ok(out)
}

fn slug(heading: &str) -> String {
    let mut slug = String::new();
    for c in heading.chars().flat_map(char::to_lowercase) {
        if c.is_alphanumeric() {
            slug.push(c);
        } else if !slug.is_empty() && !slug.ends_with('-') {
            slug.push('-');
        }
    }
    let slug = slug.trim_end_matches('-');
    if slug.is_empty() {
        "section".to_owned()
    } else {
        slug.to_owned()
    }
}

fn unique(slug: &str, used: &mut HashSet<String>) -> String {
    let mut candidate = slug.to_owned();
    let mut n = 2;
    while !used.insert(candidate.clone()) {
        candidate = format!("{slug}-{n}");
        n += 1;
    }
    candidate
}

/// GitHub-style heading anchor for fragment links.
pub fn github_anchor(heading: &str) -> String {
    let mut out = String::new();
    for c in heading.chars() {
        if c.is_whitespace() {
            if !out.ends_with('-') && !out.is_empty() {
                out.push('-');
            }
        } else if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if c == '-' || c == '_' {
            out.push(c);
        }
    }
    out.trim_matches('-').to_string()
}

#[async_trait]
impl Source for Connector {
    fn name(&self) -> &SourceName {
        &NAME
    }

    /// The snapshot is read with blocking file I/O, so it runs on the runtime's blocking pool.
    async fn fetch_all(&self) -> eros::Result<Vec<Document>> {
        let connector = self.clone();
        tokio::task::spawn_blocking(move || connector.read_snapshot()).await?
    }
}

impl Connector {
    #[eros::context("reading the markdown snapshot under {}", self.root.display())]
    fn read_snapshot(&self) -> eros::Result<Vec<Document>> {
        let mut documents = Vec::new();
        for path in self.collect_md_files()? {
            let rel = path.strip_prefix(&self.root)?;
            if let Some(standing) = self.standing(rel) {
                documents.push(self.document(&path, standing)?);
            }
        }
        Ok(documents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_front_matter() {
        let md = "---\ntitle: x\n---\n\n# Title\n\nBody";
        let sections = sections(strip_front_matter(md)).unwrap();
        assert_eq!(sections.len(), 2);
        assert!(sections[1].path.contains(&"Title".to_owned()));
    }

    #[test]
    fn ignores_headings_in_fences() {
        let md = "# Real\n\n```\n# Fake\n```\n\nMore";
        let drafts = chunks("https://example.test/a.md", md, Standing::Authoritative).unwrap();
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].heading.as_deref(), Some("Real"));
        assert!(drafts[0].text.contains("# Fake"));
    }

    #[test]
    fn nested_headings() {
        let md = "# Page\n\nIntro.\n\n## Section\n\nDetail.";
        let drafts = chunks("https://example.test/doc.md", md, Standing::Authoritative).unwrap();
        assert_eq!(drafts.len(), 2);
        assert_eq!(drafts[0].heading.as_deref(), Some("Page"));
        assert!(drafts[0].text.contains("Intro."));
        assert_eq!(drafts[1].heading.as_deref(), Some("Page > Section"));
        assert!(drafts[1].url.contains("#section"));
    }

    #[test]
    fn github_anchor_slug() {
        assert_eq!(github_anchor("Hello World!"), "hello-world");
    }

    /// A docs root removed on drop.
    struct TempDocs(PathBuf);

    impl TempDocs {
        fn new(files: &[&str]) -> Self {
            let root = std::env::temp_dir().join(format!("recall-docs-{}", uuid::Uuid::now_v7()));
            for file in files {
                let path = root.join(file);
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                fs::write(&path, format!("# {file}\n\nBody.")).unwrap();
            }
            Self(root)
        }

        fn args(&self, authoritative: &[&str], evidence: &[&str]) -> MarkdownConfig {
            MarkdownConfig {
                docs: self.0.clone(),
                base_url: None,
                authoritative: authoritative.iter().map(PathBuf::from).collect(),
                evidence: evidence.iter().map(PathBuf::from).collect(),
                skip: vec![],
            }
        }
    }

    impl Drop for TempDocs {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    async fn layers(connector: Connector) -> Vec<(String, bool)> {
        connector
            .fetch_all()
            .await
            .unwrap()
            .into_iter()
            .map(|d| {
                assert!(d.chunks.iter().all(|c| c.standing == d.chunks[0].standing));
                (d.external_id, d.chunks[0].standing.is_authoritative())
            })
            .collect()
    }

    #[tokio::test]
    async fn without_layers_the_whole_root_is_authoritative() {
        let docs = TempDocs::new(&["a.md", "sub/b.md"]);
        for (authoritative, evidence) in [(&[][..], &[][..]), (&[""][..], &[""][..])] {
            let args = docs.args(authoritative, evidence);
            let connector = Connector::new(&args).unwrap();

            assert_eq!(
                layers(connector).await,
                [("a.md".into(), true), ("sub/b.md".into(), true)]
            );
        }
    }

    #[tokio::test]
    async fn longest_listed_path_decides_and_unlisted_files_are_skipped() {
        let docs = TempDocs::new(&[
            "AGENTS.md",
            "settled/RULES.md",
            "settled-notes/other.md",
            "knowledge/notes.md",
            "knowledge/settled/ruling.md",
            "runbooks/01-setup.md",
            "runbooks/tools/README.md",
            "runbooks/tools/oracle/README.md",
            "_internal/NOTES.md",
        ]);
        let mut args = docs.args(
            &["settled", "knowledge/settled"],
            &["knowledge", "runbooks"],
        );
        args.skip = vec!["runbooks/tools".into()];
        let connector = Connector::new(&args).unwrap();

        assert_eq!(
            layers(connector).await,
            [
                ("knowledge/notes.md".into(), false),
                ("knowledge/settled/ruling.md".into(), true),
                ("runbooks/01-setup.md".into(), false),
                ("settled/RULES.md".into(), true),
            ]
        );
    }

    #[tokio::test]
    async fn base_url_joins_with_one_slash() {
        let docs = TempDocs::new(&["settled/a.md"]);
        for base in [
            "https://example.test/blob/master",
            "https://example.test/blob/master/",
        ] {
            let mut args = docs.args(&[], &[]);
            args.base_url = Some(base.into());
            let documents = Connector::new(&args).unwrap().fetch_all().await.unwrap();
            assert_eq!(
                documents[0].url,
                "https://example.test/blob/master/settled/a.md"
            );
        }
    }

    #[test]
    fn rejects_missing_escaping_and_repeated_paths() {
        let docs = TempDocs::new(&["settled/a.md"]);
        for (authoritative, evidence) in [
            (&["missing"][..], &[][..]),
            (&["../settled"][..], &[][..]),
            (&["settled"][..], &["settled"][..]),
        ] {
            assert!(Connector::new(&docs.args(authoritative, evidence)).is_err());
        }
    }
}
