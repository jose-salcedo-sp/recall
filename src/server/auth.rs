//! Per-user bearer keys from `DREW_API_KEYS`.

use std::{
    collections::{HashMap, HashSet},
    fmt,
    str::FromStr,
};

use axum::{extract::FromRequestParts, http::request::Parts};
use sha2::{Digest, Sha256};

use crate::domain::UserId;

use super::{AppState, error::ApiError};

/// Parsed from `key:user_id` pairs, comma-separated.
/// Keyed by the SHA-256 of each key, so lookup timing reveals nothing about valid keys.
#[derive(Clone)]
pub struct ApiKeys(HashMap<[u8; 32], UserId>);

impl ApiKeys {
    /// The user a bearer key maps to; `None` for an unknown key.
    pub fn user(&self, key: &str) -> Option<&UserId> {
        self.0.get(&digest(key))
    }
}

fn digest(key: &str) -> [u8; 32] {
    Sha256::digest(key.as_bytes()).into()
}

impl fmt::Debug for ApiKeys {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ApiKeys({} keys)", self.0.len())
    }
}

/// User ids that may read any run (comma-separated), for auditing Slack and API traffic.
#[derive(Clone, Default)]
pub struct AdminUsers(HashSet<UserId>);

impl AdminUsers {
    /// Whether `user` may read runs of other users.
    pub fn contains(&self, user: &UserId) -> bool {
        self.0.contains(user)
    }
}

impl fmt::Debug for AdminUsers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AdminUsers({} users)", self.0.len())
    }
}

impl FromStr for AdminUsers {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, String> {
        let users: HashSet<UserId> = value
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|id| UserId(id.into()))
            .collect();
        Ok(Self(users))
    }
}

impl FromStr for ApiKeys {
    type Err = String;

    fn from_str(pairs: &str) -> Result<Self, String> {
        let mut keys = HashMap::new();
        for pair in pairs.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let Some((key, user)) = pair.split_once(':') else {
                return Err("expected `key:user_id` pairs separated by commas".into());
            };
            let (key, user) = (key.trim(), user.trim());
            if key.is_empty() || user.is_empty() {
                return Err("API key and user id must not be empty".into());
            }
            if keys.insert(digest(key), UserId(user.into())).is_some() {
                return Err(format!("duplicate API key for user {user}"));
            }
        }
        if keys.is_empty() {
            return Err("no API keys configured".into());
        }
        Ok(Self(keys))
    }
}

/// The caller, authenticated by `Authorization: Bearer <key>`.
pub struct User(pub UserId);

impl FromRequestParts<AppState> for User {
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, state: &AppState) -> Result<Self, ApiError> {
        let key = parts
            .headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split_once(' '))
            .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
            .map(|(_, key)| key.trim())
            .ok_or_else(|| ApiError::unauthorized("Missing bearer API key."))?;
        state
            .keys
            .user(key)
            .cloned()
            .map(User)
            .ok_or_else(|| ApiError::unauthorized("Invalid API key."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_pairs() {
        let keys: ApiKeys = " k1:drew , k2:jose,".parse().unwrap();
        assert_eq!(keys.user("k1"), Some(&UserId("drew".into())));
        assert_eq!(keys.user("k2"), Some(&UserId("jose".into())));
        assert_eq!(keys.user("k3"), None);
    }

    #[test]
    fn rejects_malformed_empty_and_duplicate_keys() {
        for bad in ["", "k1", "k1:", ":drew", "k1:drew,k1:jose"] {
            assert!(bad.parse::<ApiKeys>().is_err(), "{bad:?} should not parse");
        }
    }
}
