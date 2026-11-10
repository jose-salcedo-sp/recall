//! Eval's files behind the [`Files`] port: golden sets, baselines and candidates in, results and reports out.

use std::{fs, path::Path};

/// Reads and writes whole text files.
pub trait Files: Send + Sync {
    /// The file's text.
    ///
    /// # Errors
    ///
    /// When there is no file at `path`, or it cannot be read as UTF-8.
    fn read(&self, path: &Path) -> eros::Result<String>;

    /// Creates or replaces the file at `path` with `contents`.
    ///
    /// # Errors
    ///
    /// When the file cannot be written.
    fn write(&self, path: &Path, contents: &str) -> eros::Result<()>;
}

/// The local file system.
#[derive(Clone, Copy, Debug, Default)]
pub struct LocalFiles;

impl Files for LocalFiles {
    fn read(&self, path: &Path) -> eros::Result<String> {
        read(path)
    }

    fn write(&self, path: &Path, contents: &str) -> eros::Result<()> {
        write(path, contents)
    }
}

#[eros::context("reading {}", path.display())]
fn read(path: &Path) -> eros::Result<String> {
    Ok(fs::read_to_string(path)?)
}

#[eros::context("writing {}", path.display())]
fn write(path: &Path, contents: &str) -> eros::Result<()> {
    Ok(fs::write(path, contents)?)
}

/// In-memory and failing [`Files`] for tests.
#[cfg(any(test, feature = "testing"))]
pub mod fakes {
    use std::{
        collections::BTreeMap,
        path::{Path, PathBuf},
        sync::Mutex,
    };

    use super::Files;

    /// Keeps files in a map keyed by path.
    #[derive(Debug, Default)]
    pub struct MemoryFiles(Mutex<BTreeMap<PathBuf, String>>);

    impl MemoryFiles {
        /// The file at `path`, if one was written.
        pub fn get(&self, path: impl AsRef<Path>) -> Option<String> {
            self.0
                .lock()
                .expect("lock poisoned: a test thread panicked while holding it")
                .get(path.as_ref())
                .cloned()
        }
    }

    impl Files for MemoryFiles {
        fn read(&self, path: &Path) -> eros::Result<String> {
            self.get(path).ok_or_else(|| {
                eros::error!("reading {}: no such file or directory", path.display())
            })
        }

        fn write(&self, path: &Path, contents: &str) -> eros::Result<()> {
            self.0
                .lock()
                .expect("lock poisoned: a test thread panicked while holding it")
                .insert(path.to_owned(), contents.to_owned());
            Ok(())
        }
    }

    /// Fails every call, as a full or read-only disk would, for error-path tests.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct FailingFiles;

    impl Files for FailingFiles {
        fn read(&self, path: &Path) -> eros::Result<String> {
            eros::bail!("reading {}: input/output error", path.display())
        }

        fn write(&self, path: &Path, _: &str) -> eros::Result<()> {
            eros::bail!("writing {}: no space left on device", path.display())
        }
    }
}

/// The rules every [`Files`] keeps; `files_contract!` turns them into tests, given a directory to work in.
#[cfg(any(test, feature = "testing"))]
pub mod contract {
    use std::path::Path;

    use super::Files;

    /// What is written reads back as written.
    ///
    /// # Panics
    ///
    /// When the adapter breaks the rule or fails.
    pub fn written_text_reads_back(files: &dyn Files, dir: &Path) {
        let path = dir.join("results.json");
        files
            .write(&path, "{\"ü\": 1}\n")
            .unwrap_or_else(|e| panic!("{e:#?}"));
        assert_eq!(
            files.read(&path).unwrap_or_else(|e| panic!("{e:#?}")),
            "{\"ü\": 1}\n"
        );
    }

    /// A second write replaces the first, and files at other paths are untouched.
    ///
    /// # Panics
    ///
    /// When the adapter breaks the rule or fails.
    pub fn a_write_replaces_only_its_own_file(files: &dyn Files, dir: &Path) {
        let (a, b) = (dir.join("a.json"), dir.join("b.json"));
        for (path, text) in [(&a, "first"), (&b, "other"), (&a, "second")] {
            files.write(path, text).unwrap_or_else(|e| panic!("{e:#?}"));
        }
        assert_eq!(
            files.read(&a).unwrap_or_else(|e| panic!("{e:#?}")),
            "second"
        );
        assert_eq!(files.read(&b).unwrap_or_else(|e| panic!("{e:#?}")), "other");
    }

    /// Reading a path nothing was written to is an error that names the path.
    ///
    /// # Panics
    ///
    /// When the adapter breaks the rule.
    pub fn a_missing_file_is_an_error_naming_it(files: &dyn Files, dir: &Path) {
        let path = dir.join("missing.jsonl");
        let error = files.read(&path).expect_err("no file was written");
        assert!(
            format!("{error:#?}").contains("missing.jsonl"),
            "{error:#?}"
        );
    }

    /// One test per [`Files`] check. `$make` builds the adapter and `$dir` the directory it works in.
    #[macro_export]
    macro_rules! files_contract {
        ($make:expr, $dir:expr) => {
            #[test]
            fn written_text_reads_back() {
                let dir = $dir;
                $crate::eval::files::contract::written_text_reads_back(&$make, dir.as_ref());
            }

            #[test]
            fn a_write_replaces_only_its_own_file() {
                let dir = $dir;
                $crate::eval::files::contract::a_write_replaces_only_its_own_file(
                    &$make,
                    dir.as_ref(),
                );
            }

            #[test]
            fn a_missing_file_is_an_error_naming_it() {
                let dir = $dir;
                $crate::eval::files::contract::a_missing_file_is_an_error_naming_it(
                    &$make,
                    dir.as_ref(),
                );
            }
        };
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    /// A fresh directory under the system temp dir, removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let dir = std::env::temp_dir().join(format!("recall-files-{}", uuid::Uuid::now_v7()));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl AsRef<std::path::Path> for TempDir {
        fn as_ref(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    mod local {
        crate::files_contract!(super::super::LocalFiles, super::TempDir::new());
    }

    mod memory {
        crate::files_contract!(
            super::super::fakes::MemoryFiles::default(),
            std::path::PathBuf::from("/eval")
        );
    }

    #[test]
    fn failing_files_fail() {
        use super::{Files, fakes::FailingFiles};
        let path = std::path::Path::new("/eval/out.json");
        assert!(FailingFiles.read(path).is_err());
        assert!(FailingFiles.write(path, "").is_err());
    }
}
