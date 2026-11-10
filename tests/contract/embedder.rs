//! Vectors are model output, so the contract is their shape: one `EMBEDDING_DIM` vector per input, in input order,
//! whatever the batch size.

use drew::llm::{EMBEDDING_DIM, EmbedPurpose, Embedder};
use wiremock::MockServer;

pub struct Adapter {
    pub embedder: Box<dyn Embedder>,
    pub _server: Option<MockServer>,
}

/// More texts than one OpenAI (64) or Cohere (96) batch, and than the Titan concurrency (8).
fn texts() -> Vec<String> {
    (0..100).map(|i| format!("text {i}")).collect()
}

pub async fn every_input_gets_one_vector_in_input_order(
    make: impl AsyncFnOnce() -> Adapter,
) -> eros::Result<()> {
    let adapter = make().await;
    let texts = texts();

    let batch = adapter
        .embedder
        .embed(&texts, EmbedPurpose::Document)
        .await?;

    assert_eq!(batch.len(), texts.len());
    assert!(batch.iter().all(|vector| vector.len() == EMBEDDING_DIM));
    for (text, vector) in texts.iter().zip(&batch) {
        let single = adapter
            .embedder
            .embed(std::slice::from_ref(text), EmbedPurpose::Document)
            .await?;
        assert_eq!(&single[0], vector, "{text}");
    }
    Ok(())
}

pub async fn no_inputs_give_no_vectors(make: impl AsyncFnOnce() -> Adapter) -> eros::Result<()> {
    let adapter = make().await;

    for purpose in [EmbedPurpose::Document, EmbedPurpose::Query] {
        assert!(adapter.embedder.embed(&[], purpose).await?.is_empty());
    }
    Ok(())
}

pub async fn a_query_gets_one_vector(make: impl AsyncFnOnce() -> Adapter) -> eros::Result<()> {
    let adapter = make().await;

    let vectors = adapter
        .embedder
        .embed(
            &["Which table maps users to roles?".into()],
            EmbedPurpose::Query,
        )
        .await?;

    assert_eq!(vectors.len(), 1);
    assert_eq!(vectors[0].len(), EMBEDDING_DIM);
    Ok(())
}

macro_rules! embedder_contract {
    ($make:expr) => {
        #[tokio::test]
        async fn every_input_gets_one_vector_in_input_order() -> eros::Result<()> {
            crate::embedder::every_input_gets_one_vector_in_input_order($make).await
        }

        #[tokio::test]
        async fn no_inputs_give_no_vectors() -> eros::Result<()> {
            crate::embedder::no_inputs_give_no_vectors($make).await
        }

        #[tokio::test]
        async fn a_query_gets_one_vector() -> eros::Result<()> {
            crate::embedder::a_query_gets_one_vector($make).await
        }
    };
}

/// A vector that depends only on `text`, as a real model's does.
fn vector_of(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0; EMBEDDING_DIM];
    for (i, byte) in text.bytes().enumerate() {
        vector[i % EMBEDDING_DIM] += f32::from(byte);
    }
    vector
}

mod hash {
    use drew::llm::fakes::HashEmbedder;

    use super::Adapter;

    async fn hash() -> Adapter {
        Adapter {
            embedder: Box::new(HashEmbedder),
            _server: None,
        }
    }

    embedder_contract!(hash);
}

mod openai {
    use std::num::NonZeroUsize;

    use drew::llm::{EMBEDDING_DIM, EmbedPurpose, Embedder, openai::OpenAiEmbedder};
    use serde_json::{Value, json};
    use wiremock::{
        Mock, MockServer, Request, ResponseTemplate,
        matchers::{method, path},
    };

    use super::{Adapter, vector_of};

    /// Answers each batch with its vectors in reverse order, each tagged with its input index.
    fn embeddings(request: &Request) -> ResponseTemplate {
        let body: Value = request.body_json().unwrap();
        let mut data: Vec<_> = body["input"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, text)| {
                json!({ "index": index, "embedding": vector_of(text.as_str().unwrap()) })
            })
            .collect();
        data.reverse();
        ResponseTemplate::new(200).set_body_json(json!({ "object": "list", "data": data }))
    }

    async fn server(response: impl wiremock::Respond + 'static) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(response)
            .mount(&server)
            .await;
        server
    }

    fn embedder(server: &MockServer) -> OpenAiEmbedder {
        OpenAiEmbedder::new(&format!("{}/v1", server.uri()).parse().unwrap(), "bge-m3").unwrap()
    }

    async fn openai() -> Adapter {
        let server = server(embeddings).await;
        Adapter {
            embedder: Box::new(embedder(&server)),
            _server: Some(server),
        }
    }

    embedder_contract!(openai);

    mod workers_ai {
        use drew::llm::{
            ModelRef,
            workers_ai::{WorkersAiEmbedModel, WorkersAiEmbedder},
        };
        use wiremock::Respond;

        use super::*;

        const ROUTE: &str = "/client/v4/accounts/acc-123/ai/run/@cf/baai/bge-m3";
        const BGE_M3_LIMIT: usize = 60_000;
        const BGE_M3_INPUT_LIMIT: usize = 8_192;

        fn failure(message: &str) -> ResponseTemplate {
            ResponseTemplate::new(400).set_body_json(json!({
                "result": {},
                "success": false,
                "errors": [{ "code": 5006, "message": message }],
                "messages": [],
            }))
        }

        /// The texts of each native `ai/run` request, `{"text": [...], "truncate_inputs": true}`.
        fn sent_texts(requests: &[Request]) -> Vec<Vec<String>> {
            requests
                .iter()
                .map(|request| {
                    let body: Value = request.body_json().unwrap();
                    assert_eq!(body["truncate_inputs"], true, "{body}");
                    body["text"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|text| text.as_str().unwrap().to_owned())
                        .collect()
                })
                .collect()
        }

        /// Workers AI's native bge-m3 route, at 3 bytes a token: without `truncate_inputs` an input over 8,192 tokens
        /// fails, with it the input is cut to 8,192; then a request counts as its input count times its longest
        /// input, as it did for `canon/LIMITS.md`. Answers in the envelope, `{"shape": [n, 1024], "data": [...]}`.
        struct BgeM3;

        impl Respond for BgeM3 {
            fn respond(&self, request: &Request) -> ResponseTemplate {
                let body: Value = request.body_json().unwrap();
                let texts: Vec<&str> = body["text"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|text| text.as_str().unwrap())
                    .collect();
                let truncate = body["truncate_inputs"] == true;
                let mut longest = 0;
                for text in &texts {
                    let tokens = text.len().div_ceil(3);
                    if tokens > BGE_M3_INPUT_LIMIT && !truncate {
                        return failure(&format!(
                            "Sequence too long: {tokens} > {BGE_M3_INPUT_LIMIT}"
                        ));
                    }
                    longest = longest.max(tokens.min(BGE_M3_INPUT_LIMIT));
                }
                let tokens = texts.len() * longest;
                if tokens > BGE_M3_LIMIT {
                    return failure(&format!(
                        "Max context reached {tokens} tokens but model supports only {BGE_M3_LIMIT}"
                    ));
                }
                let data: Vec<_> = texts.iter().map(|text| vector_of(text)).collect();
                ResponseTemplate::new(200).set_body_json(json!({
                    "result": { "shape": [texts.len(), EMBEDDING_DIM], "data": data },
                    "success": true,
                    "errors": [],
                    "messages": [],
                }))
            }
        }

        async fn responding(response: impl Respond + 'static) -> MockServer {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path(ROUTE))
                .respond_with(response)
                .mount(&server)
                .await;
            server
        }

        fn bge_m3(server: &MockServer, key: &str) -> WorkersAiEmbedder {
            WorkersAiEmbedder::new(
                "acc-123".parse().unwrap(),
                WorkersAiEmbedModel::BgeM3,
                key.parse().unwrap(),
            )
            .unwrap()
            .with_api_base(&format!("{}/client/v4", server.uri()).parse().unwrap())
        }

        async fn workers_ai() -> Adapter {
            let server = responding(BgeM3).await;
            Adapter {
                embedder: Box::new(bge_m3(&server, "cf-token")),
                _server: Some(server),
            }
        }

        embedder_contract!(workers_ai);

        /// A section of `canon/LIMITS.md`'s shape: a heading and a table dense in digits.
        fn section(i: usize, rows: usize) -> String {
            format!(
                "LIMITS\nLimit {i}\n\n{}",
                "| api | 1,200/min |\n".repeat(rows)
            )
        }

        #[tokio::test]
        async fn requests_stay_under_the_models_padded_limit_with_no_setting() {
            let server = responding(BgeM3).await;
            let mut texts: Vec<String> = (0..35).map(|i| section(i, 3)).collect();
            texts[20] = section(20, 436);
            texts.extend((35..135).map(|i| section(i, 50 + (i * 37) % 1_400)));
            assert!(texts[20].len() > 8_700);
            assert!(texts.iter().any(|text| text.len() > 27_000));
            assert!(padded(&texts[..35]) > BGE_M3_LIMIT);

            let vectors = bge_m3(&server, "cf-token")
                .embed(&texts, EmbedPurpose::Document)
                .await
                .unwrap();

            let sent = sent_texts(&server.received_requests().await.unwrap());
            for batch in &sent {
                assert!(padded(batch) <= 50_000, "{} texts", batch.len());
                assert!(batch.len() <= 64);
            }
            assert_eq!(sent.concat(), texts);
            let expected: Vec<_> = texts.iter().map(|text| vector_of(text)).collect();
            assert_eq!(vectors, expected);
        }

        /// `A1-PASTEBACK.md` failed on the OpenAI-compatible route with `Sequence too long: 10670 > 8192`.
        #[tokio::test]
        async fn an_input_over_the_models_input_limit_is_embedded_not_rejected() {
            let server = responding(BgeM3).await;
            let texts = [
                section(0, 3),
                section(1, 1_800),
                section(2, 3),
                section(3, 8_000),
            ];
            assert!(texts[1].len().div_ceil(3) > 12_000);
            assert!(texts[3].len().div_ceil(3) > 50_000);

            let vectors = bge_m3(&server, "cf-token")
                .embed(&texts, EmbedPurpose::Document)
                .await
                .unwrap();

            let expected: Vec<_> = texts.iter().map(|text| vector_of(text)).collect();
            assert_eq!(vectors, expected);
            let sent = sent_texts(&server.received_requests().await.unwrap());
            assert!(
                sent.contains(&vec![texts[3].clone()]),
                "over the budget, alone"
            );
            assert_eq!(sent.concat(), texts);
        }

        #[tokio::test]
        async fn the_token_goes_to_the_accounts_route_and_chunks_keep_their_model() {
            let server = responding(BgeM3).await;
            let embedder = bge_m3(&server, "cf-token");

            embedder
                .embed(&["a".into()], EmbedPurpose::Query)
                .await
                .unwrap();

            let sent = &server.received_requests().await.unwrap()[0];
            assert_eq!(sent.url.path(), ROUTE);
            assert_eq!(
                sent.headers.get("authorization").unwrap(),
                "Bearer cf-token"
            );
            assert_eq!(
                sent.body_json::<Value>().unwrap(),
                json!({ "text": ["a"], "truncate_inputs": true })
            );
            let openai_compatible = super::embedder(&server).model();
            assert_eq!(
                embedder.model(),
                ModelRef::new("@cf/baai/bge-m3", openai_compatible.version)
            );
        }

        #[tokio::test]
        async fn an_error_envelope_names_the_model_and_its_errors_but_not_the_token() {
            let server = responding(failure("Sequence too long: 10670 > 8192")).await;

            let error = bge_m3(&server, "cf-secret-token")
                .embed(&["a".into()], EmbedPurpose::Document)
                .await
                .unwrap_err();

            let report = format!("{error:?}");
            assert!(
                report.contains(
                    "@cf/baai/bge-m3 failed with 400 Bad Request: 5006: Sequence too long: 10670 > 8192"
                ) && report.contains(ROUTE),
                "{report}"
            );
            assert!(!report.contains("cf-secret-token"), "{report}");
        }

        #[tokio::test]
        async fn a_body_outside_the_envelope_names_its_host_and_quotes_200_bytes() {
            let body = format!("<html>bad gateway {}", "x".repeat(300));
            let server = responding(ResponseTemplate::new(502).set_body_string(body.clone())).await;

            let error = bge_m3(&server, "cf-token")
                .embed(&["a".into()], EmbedPurpose::Document)
                .await
                .unwrap_err();

            let report = format!("{error:#?}");
            let host = server.address().to_string();
            assert!(
                report.contains(&format!(
                    "@cf/baai/bge-m3 at {host} returned 502 Bad Gateway without the Workers AI envelope"
                )) && report.contains(&body[..200])
                    && !report.contains(&body[..201]),
                "{report}"
            );
        }

        #[tokio::test]
        async fn fewer_vectors_than_inputs_is_an_error() {
            let server = responding(ResponseTemplate::new(200).set_body_json(json!({
                "result": { "shape": [1, EMBEDDING_DIM], "data": [vec![0.0; EMBEDDING_DIM]] },
                "success": true, "errors": [], "messages": [],
            })))
            .await;

            let error = bge_m3(&server, "cf-token")
                .embed(&["a".into(), "b".into()], EmbedPurpose::Document)
                .await
                .unwrap_err();

            assert!(
                format!("{error:#?}").contains("expected 2 embeddings, got 1"),
                "{error:#?}"
            );
        }
    }

    mod token_budget {
        use super::*;

        /// A budget of about six of the contract's texts a request, so the suite's 100 texts span many batches.
        async fn budgeted() -> Adapter {
            let server = server(embeddings).await;
            Adapter {
                embedder: Box::new(
                    embedder(&server).with_max_batch_tokens(NonZeroUsize::new(BUDGET)),
                ),
                _server: Some(server),
            }
        }

        embedder_contract!(budgeted);
    }

    const BUDGET: usize = 20;

    fn sent_inputs(requests: &[Request]) -> Vec<Vec<String>> {
        requests
            .iter()
            .map(|request| {
                request.body_json::<Value>().unwrap()["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|text| text.as_str().unwrap().to_owned())
                    .collect()
            })
            .collect()
    }

    #[tokio::test]
    async fn a_budget_packs_each_request_under_it_in_input_order() {
        let server = server(embeddings).await;
        let texts: Vec<String> = (0..40).map(|i| format!("text {i}")).collect();

        let vectors = embedder(&server)
            .with_max_batch_tokens(NonZeroUsize::new(BUDGET))
            .embed(&texts, EmbedPurpose::Document)
            .await
            .unwrap();

        let sent = sent_inputs(&server.received_requests().await.unwrap());
        assert!(sent.len() > 1, "{sent:?}");
        for batch in &sent {
            assert!(padded(batch) <= BUDGET, "{batch:?}");
        }
        assert_eq!(sent.concat(), texts);
        let expected: Vec<_> = texts.iter().map(|text| vector_of(text)).collect();
        assert_eq!(vectors, expected);
    }

    /// A request's estimated tokens as Workers AI counts them: every text padded to the longest, at 3 bytes a token.
    fn padded(batch: &[String]) -> usize {
        batch.len()
            * batch
                .iter()
                .map(|text| text.len().div_ceil(3))
                .max()
                .unwrap_or(0)
    }

    /// `canon/LIMITS.md`'s shape: 35 sections, one of about 8,770 bytes. Sent as one request they failed with `Max
    /// context reached 102060 tokens but model supports only 60000` (35 x 2,916), though they sum to far less.
    #[tokio::test]
    async fn one_long_section_among_many_short_ones_splits_the_request() {
        let server = server(embeddings).await;
        let mut texts: Vec<String> = (0..35)
            .map(|i| format!("LIMITS\nLimit {i}\n\nOne short rule."))
            .collect();
        texts[20] = format!(
            "LIMITS\nRate limits\n\n{}",
            "| api | 1,200/min |\n".repeat(436)
        );
        assert!(texts[20].len() > 8_700);
        assert!(padded(&texts) > 60_000);

        let vectors = embedder(&server)
            .with_max_batch_tokens(NonZeroUsize::new(50_000))
            .embed(&texts, EmbedPurpose::Document)
            .await
            .unwrap();

        let sent = sent_inputs(&server.received_requests().await.unwrap());
        assert!(sent.len() > 1, "{} requests", sent.len());
        for batch in &sent {
            assert!(padded(batch) <= 50_000, "{} texts", batch.len());
        }
        assert_eq!(sent.concat(), texts);
        assert_eq!(vectors.len(), texts.len());
    }

    #[tokio::test]
    async fn a_text_over_the_budget_fails_before_any_request() {
        let server = server(embeddings).await;
        let texts = [
            "Limits\nRate limits\n\nshort".to_owned(),
            format!("LIMITS\nEvery limit\n\n{}", "x".repeat(200)),
        ];

        let error = embedder(&server)
            .with_max_batch_tokens(NonZeroUsize::new(BUDGET))
            .embed(&texts, EmbedPurpose::Document)
            .await
            .unwrap_err();

        let report = format!("{error:#?}");
        assert!(
            report.contains("text 1 of 2")
                && report.contains("LIMITS\\nEvery limit")
                && report.contains("DREW_EMBED_MAX_BATCH_TOKENS"),
            "{report}"
        );
        assert!(server.received_requests().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn the_key_is_sent_as_a_bearer_token_only_when_set() {
        for key in [Some("cf-token"), None] {
            let server = server(embeddings).await;

            embedder(&server)
                .with_key(key.map(|key| key.parse().unwrap()))
                .embed(&["a".into()], EmbedPurpose::Query)
                .await
                .unwrap();

            let sent = server.received_requests().await.unwrap();
            let authorization = sent[0]
                .headers
                .get("authorization")
                .map(|value| value.to_str().unwrap().to_owned());
            assert_eq!(authorization, key.map(|key| format!("Bearer {key}")));
        }
    }

    #[tokio::test]
    async fn texts_go_in_batches_of_64() {
        let server = server(embeddings).await;
        let texts: Vec<String> = (0..65).map(|i| format!("text {i}")).collect();

        embedder(&server)
            .embed(&texts, EmbedPurpose::Document)
            .await
            .unwrap();

        let sizes: Vec<_> = server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .map(|request| {
                request.body_json::<Value>().unwrap()["input"]
                    .as_array()
                    .unwrap()
                    .len()
            })
            .collect();
        assert_eq!(sizes, [64, 1]);
    }

    #[tokio::test]
    async fn fewer_vectors_than_inputs_is_an_error() {
        let server = server(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{ "index": 0, "embedding": vec![0.0; EMBEDDING_DIM] }]
        })))
        .await;

        let error = embedder(&server)
            .embed(&["a".into(), "b".into()], EmbedPurpose::Document)
            .await
            .unwrap_err();

        assert!(
            format!("{error:#?}").contains("expected 2 embeddings, got 1"),
            "{error:#?}"
        );
    }

    #[tokio::test]
    async fn an_error_status_carries_the_status_and_body() {
        let server =
            server(ResponseTemplate::new(404).set_body_string("model \"bge-m3\" not found")).await;

        let error = embedder(&server)
            .embed(&["a".into()], EmbedPurpose::Document)
            .await
            .unwrap_err();

        let report = format!("{error:#?}");
        assert!(
            report.contains("404") && report.contains("not found"),
            "{report}"
        );
    }
}

mod bedrock {
    use std::time::Duration;

    use drew::llm::{EMBEDDING_DIM, EmbedPurpose, Embedder, bedrock_embed::BedrockEmbedder};
    use serde_json::{Value, json};
    use wiremock::{
        Mock, MockServer, Request, ResponseTemplate,
        matchers::{method, path_regex},
    };

    use super::{Adapter, vector_of};

    const TITAN: &str = "amazon.titan-embed-text-v2:0";
    const COHERE: &str = "cohere.embed-english-v3";

    /// Titan embeds one text per call. Later texts answer sooner, so out-of-order completion shows.
    fn titan(request: &Request) -> ResponseTemplate {
        let body: Value = request.body_json().unwrap();
        let text = body["inputText"].as_str().unwrap();
        let n: u64 = text
            .rsplit(' ')
            .next()
            .and_then(|n| n.parse().ok())
            .unwrap_or(0);
        ResponseTemplate::new(200)
            .set_delay(Duration::from_millis(5 * (7 - n % 8)))
            .set_body_json(json!({ "embedding": vector_of(text), "inputTextTokenCount": 2 }))
    }

    fn cohere(request: &Request) -> ResponseTemplate {
        let body: Value = request.body_json().unwrap();
        let texts = body["texts"].as_array().unwrap();
        let embeddings: Vec<_> = texts
            .iter()
            .map(|text| vector_of(text.as_str().unwrap()))
            .collect();
        ResponseTemplate::new(200).set_body_json(json!({
            "id": "test", "embeddings": embeddings, "texts": texts, "response_type": "embeddings_floats"
        }))
    }

    async fn server(response: impl wiremock::Respond + 'static) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex("^/model/[^/]+/invoke$"))
            .respond_with(response)
            .mount(&server)
            .await;
        server
    }

    fn embedder(server: &MockServer, model: &str) -> BedrockEmbedder {
        BedrockEmbedder::new(crate::bedrock::client(server), model).unwrap()
    }

    mod titan {
        use super::*;

        async fn titan_v2() -> Adapter {
            let server = server(titan).await;
            Adapter {
                embedder: Box::new(embedder(&server, TITAN)),
                _server: Some(server),
            }
        }

        embedder_contract!(titan_v2);
    }

    mod cohere {
        use super::*;

        async fn cohere_v3() -> Adapter {
            let server = server(cohere).await;
            Adapter {
                embedder: Box::new(embedder(&server, COHERE)),
                _server: Some(server),
            }
        }

        embedder_contract!(cohere_v3);
    }

    #[tokio::test]
    async fn a_rejected_call_names_the_model_and_the_reason() {
        let server = server(
            ResponseTemplate::new(400)
                .insert_header("x-amzn-errortype", "ValidationException")
                .set_body_json(json!({ "message": "The provided model identifier is invalid." })),
        )
        .await;

        let error = embedder(&server, TITAN)
            .embed(&["a".into()], EmbedPurpose::Document)
            .await
            .unwrap_err();

        let report = format!("{error:#?}");
        assert!(
            report.contains(TITAN) && report.contains("model identifier is invalid"),
            "{report}"
        );
    }

    #[tokio::test]
    async fn a_wrong_dimension_is_an_error() {
        let server = server(
            ResponseTemplate::new(200).set_body_json(json!({ "embedding": vec![0.0; 768] })),
        )
        .await;

        let error = embedder(&server, TITAN)
            .embed(&["a".into()], EmbedPurpose::Document)
            .await
            .unwrap_err();

        assert!(
            format!("{error:#?}").contains(&format!("expected {EMBEDDING_DIM}-dimensional")),
            "{error:#?}"
        );
    }

    #[tokio::test]
    async fn cohere_sends_the_purpose_as_its_input_type() {
        let server = server(cohere).await;
        let embedder = embedder(&server, COHERE);

        embedder
            .embed(&["a".into()], EmbedPurpose::Document)
            .await
            .unwrap();
        embedder
            .embed(&["a".into()], EmbedPurpose::Query)
            .await
            .unwrap();

        let types: Vec<_> = server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .map(|request| request.body_json::<Value>().unwrap()["input_type"].clone())
            .collect();
        assert_eq!(types, ["search_document", "search_query"]);
    }
}
