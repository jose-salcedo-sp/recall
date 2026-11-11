//! The `Source` contract (`recall::ingest::contract`) on the shipped markdown connector and on the in-memory fake.

use std::{fs, path::PathBuf};

use async_trait::async_trait;
use chrono::{TimeZone, Utc};
use recall::{
    domain::{ChunkDraft, Document, Labels, Sensitivity, SourceName},
    ingest::{
        Source,
        fakes::StaticSource,
        markdown::{Connector, MarkdownConfig},
    },
};

fn fake() -> StaticSource {
    let name = SourceName::new("wiki").unwrap();
    let page = |external_id: &str, locators: &[&str]| Document {
        source: name.clone(),
        external_id: external_id.into(),
        title: external_id.to_uppercase(),
        url: format!("https://wiki.example.test/{external_id}"),
        updated_at: Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
        labels: Labels {
            acl: recall::access::Acl::anyone(),
            tenant: None,
            sensitivity: Sensitivity::Internal,
        },
        chunks: locators
            .iter()
            .map(|locator| ChunkDraft {
                locator: (*locator).into(),
                heading: Some((*locator).into()),
                url: format!("https://wiki.example.test/{external_id}#{locator}"),
                text: format!("{locator} of {external_id}"),
                authoritative: true,
            })
            .collect(),
    };
    StaticSource {
        documents: vec![page("home", &["intro", "roles"]), page("faq", &["intro"])],
        name,
    }
}

/// A markdown connector over a docs root it creates, and removes on drop.
struct MarkdownDocs {
    root: PathBuf,
    connector: Connector,
}

impl MarkdownDocs {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("recall-source-{}", uuid::Uuid::now_v7()));
        for (file, text) in [
            (
                "roles.md",
                "# Roles\n\nIntro.\n\n## Owners\n\nOwners.\n\n## Owners\n\nAgain.",
            ),
            (
                "notes/leavers.md",
                "Leavers are locked first.\n\n## Empty\n",
            ),
            (".drafts/skip.md", "# Not in the corpus"),
        ] {
            let path = root.join(file);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, text).unwrap();
        }
        let connector = Connector::new(&MarkdownConfig {
            docs: root.clone(),
            base_url: Some("https://docs.example.test/blob/main".into()),
            ..MarkdownConfig::default()
        })
        .unwrap();
        Self { root, connector }
    }
}

impl Drop for MarkdownDocs {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[async_trait]
impl Source for MarkdownDocs {
    fn name(&self) -> &SourceName {
        self.connector.name()
    }

    async fn fetch_all(&self) -> eros::Result<Vec<Document>> {
        self.connector.fetch_all().await
    }
}

mod static_source {
    recall::source_contract!(#[tokio::test] () => super::fake());
}

mod markdown {
    recall::source_contract!(#[tokio::test] () => super::MarkdownDocs::new());
}
