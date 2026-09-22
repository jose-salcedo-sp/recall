//! recall-dash — live TUI over the service's JSON log stream.
//!
//! Reads what the service already emits instead of adding an endpoint or a metrics
//! channel: every number here comes from existing `tracing` events.
//!
//!   recall-dash /tmp/recall.jsonl          # preferred: follow the dev.sh log file
//!   LOG_FORMAT=json ./recall 2>&1 | recall-dash   # pipe mode (macOS needs use-dev-tty)
//!
//! j / k moves the request list. Enter opens that request: every pipeline
//! stage, what it chose, and the text that was published. Esc backs out.
//! q / Ctrl-C quits.

use std::collections::{HashMap, VecDeque};
use std::io;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::prelude::*;
use ratatui::widgets::*;
use serde_json::Value;

const STAGES: [&str; 6] = ["kind", "embed", "retrieve", "admit", "generate", "verify"];
const SPARK_WIDTH: usize = 120;

#[derive(Default)]
struct Stage {
    samples: VecDeque<u64>,
    fails: u64,
}

impl Stage {
    fn push(&mut self, ms: u64, ok: bool) {
        self.samples.push_back(ms);
        if self.samples.len() > SPARK_WIDTH {
            self.samples.pop_front();
        }
        if !ok {
            self.fails += 1;
        }
    }
    fn last(&self) -> u64 {
        self.samples.back().copied().unwrap_or(0)
    }
    fn max(&self) -> u64 {
        self.samples.iter().copied().max().unwrap_or(0)
    }
    /// p50 by sorting a copy — the window is 120 samples, so this is free.
    fn p50(&self) -> u64 {
        if self.samples.is_empty() {
            return 0;
        }
        let mut v: Vec<u64> = self.samples.iter().copied().collect();
        v.sort_unstable();
        v[v.len() / 2]
    }
}

struct StageHit {
    name: String,
    ms: u64,
    ok: bool,
    error: String,
}

struct VerdictRow {
    claim: String,
    verdict: String,
    memory_index: Option<u64>,
}

struct VerifyRound {
    attempt: u64,
    all_ok: bool,
    claims: u64,
    supported: u64,
    verdicts: Vec<VerdictRow>,
}

struct CandidateRow {
    id: String,
    statement: String,
    text: String,
    noul: f64,
    origin: String,
    admitted: bool,
    injection: Option<f64>,
    contradicts: Option<f64>,
    relevant: Option<f64>,
    evidence: Option<f64>,
    route: String,
    rrf: Option<f64>,
    source: String,
    grantor: String,
}

struct Ask {
    id: String,
    question: String,
    brain_id: String,
    as_of: String,
    personal: u64,
    granted: u64,
    retrieve_k: u64,
    merged: u64,
    top_rrf: f64,
    admitted: u64,
    candidates: u64,
    conflicts: u64,
    top_noul: f64,
    threshold: f64,
    calibrated: bool,
    empty: bool,
    total_ms: u64,
    ranked: Vec<CandidateRow>,
    complete: bool,
    failed: bool,
    kind: String,
    kind_confidence: Option<f64>,
    model_choice: String,
    fell_back: Option<bool>,
    chitchat: bool,
    embed_dims: Option<u64>,
    stages: Vec<StageHit>,
    drafts: Vec<String>,
    generate_citations: Option<u64>,
    verify_rounds: Vec<VerifyRound>,
    regens: u64,
    published: Option<String>,
    skip_generate: bool,
}

#[derive(Default)]
struct App {
    stages: HashMap<String, Stage>,
    asks: VecDeque<Ask>,
    events: VecDeque<(String, String)>,
    total_asks: u64,
    empty_admits: u64,
    errors: u64,
    threshold: f64,
    calibrated: bool,
    started: Option<Instant>,
    /// Index into `asks` (0 = oldest). j/k moves selection.
    selected: Option<usize>,
    /// Enter opens the pipeline for `selected`. Esc closes it.
    inspect: bool,
    scroll: u16,
    view_rows: usize,
}

impl App {
    fn stage(&mut self, name: &str) -> &mut Stage {
        self.stages.entry(name.to_string()).or_default()
    }

    fn ensure_ask(&mut self, id: &str) -> &mut Ask {
        if let Some(i) = self
            .asks
            .iter()
            .position(|a| a.id == id || a.id.starts_with(id) || id.starts_with(&a.id))
        {
            return &mut self.asks[i];
        }
        self.asks.push_back(Ask {
            id: id.to_string(),
            ..Default::default()
        });
        if self.asks.len() > 200 {
            self.asks.pop_front();
            self.selected = self.selected.map(|s| s.saturating_sub(1));
        }
        let idx = self.asks.len() - 1;
        self.follow(idx);
        self.asks.back_mut().unwrap()
    }

    fn ingest(&mut self, line: &str) {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            return;
        };
        let msg = v.get("message").and_then(Value::as_str).unwrap_or("");
        let num = |k: &str| v.get(k).and_then(Value::as_f64);
        let text = |k: &str| v.get(k).and_then(Value::as_str).map(str::to_string);

        match msg {
            "ask started" => {
                let id = text("ask_id").unwrap_or_default();
                if id.is_empty() {
                    return;
                }
                let ask = self.ensure_ask(&id);
                ask.question = text("question").unwrap_or_default();
                ask.brain_id = text("brain_id").unwrap_or_default();
                ask.as_of = tidy_debug_opt(&text("as_of").unwrap_or_default());
            }
            "kind chosen" => {
                let id = text("ask_id").unwrap_or_default();
                if id.is_empty() {
                    return;
                }
                let kind = text("kind").unwrap_or_default();
                let model_choice = text("model_choice").unwrap_or_default();
                let confidence = num("confidence");
                let fell_back = v.get("fell_back").and_then(Value::as_bool);
                let chitchat = v.get("chitchat").and_then(Value::as_bool).unwrap_or(false);
                let ask = self.ensure_ask(&id);
                ask.kind = kind;
                ask.kind_confidence = confidence;
                ask.model_choice = model_choice;
                ask.fell_back = fell_back;
                if chitchat {
                    ask.chitchat = true;
                }
            }
            "embedded" => {
                let id = text("ask_id").unwrap_or_default();
                if id.is_empty() {
                    return;
                }
                let dims = num("dims").map(|n| n as u64);
                let ask = self.ensure_ask(&id);
                ask.embed_dims = dims;
            }
            "retrieved from nexus" => {
                let id = text("ask_id").unwrap_or_default();
                if id.is_empty() {
                    return;
                }
                let personal = num("personal").unwrap_or(0.0) as u64;
                let granted = num("granted").unwrap_or(0.0) as u64;
                let k = num("k").unwrap_or(0.0) as u64;
                let ask = self.ensure_ask(&id);
                ask.personal = personal;
                ask.granted = granted;
                ask.retrieve_k = k;
            }
            "retrieve merged" => {
                let id = text("ask_id").unwrap_or_default();
                if id.is_empty() {
                    return;
                }
                let merged = num("merged").unwrap_or(0.0) as u64;
                let top_rrf = num("top_rrf").unwrap_or(0.0);
                let ask = self.ensure_ask(&id);
                ask.merged = merged;
                ask.top_rrf = top_rrf;
            }
            "stage ok" | "stage failed" => {
                let (Some(stage), Some(ms)) = (text("stage"), num("ms")) else {
                    return;
                };
                let ok = msg == "stage ok";
                let err = text("error").unwrap_or_default();
                self.stage(&stage).push(ms as u64, ok);
                if let Some(id) = text("ask_id").filter(|id| !id.is_empty()) {
                    let ask = self.ensure_ask(&id);
                    ask.stages.push(StageHit {
                        name: stage.clone(),
                        ms: ms as u64,
                        ok,
                        error: err.clone(),
                    });
                    ask.total_ms = ask.stages.iter().map(|s| s.ms).sum();
                    if !ok {
                        ask.failed = true;
                    }
                }
                if !ok {
                    self.errors += 1;
                    self.events.push_back((format!("{stage} FAILED"), err));
                }
            }
            "admission complete" => {
                let id = text("ask_id").unwrap_or_default();
                let admitted = num("admitted").unwrap_or(0.0) as u64;
                self.threshold = num("threshold").unwrap_or(self.threshold);
                self.calibrated = v
                    .get("calibrated")
                    .and_then(Value::as_bool)
                    .unwrap_or(self.calibrated);
                let question = text("question").unwrap_or_default();
                let ranked = parse_ranked(v.get("ranked"));
                let conflicts = num("conflicts").unwrap_or(0.0) as u64;
                let candidates = num("candidates").unwrap_or(0.0) as u64;
                let top_noul = num("top_noul").unwrap_or(0.0);
                let threshold = self.threshold;
                let calibrated = self.calibrated;
                let idx = if id.is_empty() {
                    self.asks.push_back(Ask::default());
                    self.asks.len() - 1
                } else if let Some(i) = self
                    .asks
                    .iter()
                    .position(|a| a.id == id || a.id.starts_with(&id) || id.starts_with(&a.id))
                {
                    i
                } else {
                    self.asks.push_back(Ask {
                        id: id.clone(),
                        ..Default::default()
                    });
                    self.asks.len() - 1
                };
                let ask = &mut self.asks[idx];
                if !ask.complete {
                    self.total_asks += 1;
                    if admitted == 0 {
                        self.empty_admits += 1;
                    }
                }
                if ask.question.is_empty() {
                    ask.question = question;
                }
                ask.admitted = admitted;
                ask.candidates = candidates;
                ask.top_noul = top_noul;
                ask.empty = admitted == 0;
                ask.conflicts = conflicts;
                ask.threshold = threshold;
                ask.calibrated = calibrated;
                if !ranked.is_empty() {
                    ask.ranked = ranked;
                }
                ask.complete = true;
                self.follow(idx);
                if self.asks.len() > 200 {
                    self.asks.pop_front();
                    self.selected = self.selected.map(|s| s.saturating_sub(1));
                }
            }
            // Emitted separately from "admission complete" because it carries the
            // question and memory text, so it is debug-only.
            "ask detail" => {
                let id = text("ask_id").unwrap_or_default();
                if id.is_empty() {
                    return;
                }
                let (q, ranked) = (text("question"), parse_ranked(v.get("ranked")));
                let ask = self.ensure_ask(&id);
                if let Some(q) = q {
                    ask.question = q;
                }
                if !ranked.is_empty() {
                    ask.ranked = ranked;
                }
            }
            "generated" => {
                let id = text("ask_id").unwrap_or_default();
                if id.is_empty() {
                    return;
                }
                let chitchat = v.get("chitchat").and_then(Value::as_bool).unwrap_or(false);
                let citations = num("citations").map(|n| n as u64);
                let answer = text("answer");
                let ask = self.ensure_ask(&id);
                if chitchat {
                    ask.chitchat = true;
                }
                if let Some(n) = citations {
                    ask.generate_citations = Some(n);
                }
                if let Some(answer) = answer {
                    ask.drafts.push(answer);
                }
            }
            "verify verdicts" => {
                let id = text("ask_id").unwrap_or_default();
                if id.is_empty() {
                    return;
                }
                let round = VerifyRound {
                    attempt: num("attempt").unwrap_or(0.0) as u64,
                    all_ok: v.get("all_ok").and_then(Value::as_bool).unwrap_or(false),
                    claims: num("claims").unwrap_or(0.0) as u64,
                    supported: num("supported").unwrap_or(0.0) as u64,
                    verdicts: parse_verdicts(v.get("verdicts")),
                };
                self.ensure_ask(&id).verify_rounds.push(round);
            }
            "answer published" => {
                let id = text("ask_id").unwrap_or_default();
                if id.is_empty() {
                    return;
                }
                let empty = v.get("empty").and_then(Value::as_bool).unwrap_or(false);
                let chitchat = v.get("chitchat").and_then(Value::as_bool).unwrap_or(false);
                let answer = text("answer");
                let ask = self.ensure_ask(&id);
                if chitchat {
                    ask.chitchat = true;
                }
                let was_complete = ask.complete;
                let was_empty = ask.empty;
                ask.published = answer;
                ask.empty = empty;
                ask.complete = true;
                if !was_complete {
                    self.total_asks += 1;
                    if empty {
                        self.empty_admits += 1;
                    }
                } else if empty && !was_empty {
                    self.empty_admits += 1;
                }
            }
            "empty admission; skipping generator" => {
                let id = text("ask_id").unwrap_or_default();
                if id.is_empty() {
                    return;
                }
                self.ensure_ask(&id).skip_generate = true;
            }
            "verify failed; regenerating once" => {
                let attempt = num("attempt").unwrap_or(1.0);
                if let Some(id) = text("ask_id").filter(|id| !id.is_empty()) {
                    let ask = self.ensure_ask(&id);
                    ask.regens = ask.regens.max(attempt as u64);
                }
                self.events
                    .push_back(("REGEN".into(), format!("verify attempt {attempt}")));
            }
            "withheld secret-sensitivity rows" => self.events.push_back((
                "WITHHELD".into(),
                format!("{} secret rows", num("withheld").unwrap_or(0.0)),
            )),
            "retrying once" => self
                .events
                .push_back(("RETRY".into(), text("stage").unwrap_or_default())),
            m if m.starts_with("failed to persist") => {
                self.errors += 1;
                self.events.push_back(("PERSIST".into(), m.into()));
            }
            _ => {}
        }
        while self.events.len() > 100 {
            self.events.pop_front();
        }
    }

    fn move_selection(&mut self, delta: i32) {
        if self.asks.is_empty() {
            return;
        }
        let n = self.asks.len();
        let cur = self.selected.unwrap_or(n.saturating_sub(1));
        let next = (cur as i32 + delta).clamp(0, n as i32 - 1) as usize;
        self.selected = Some(next);
    }

    /// Live tail follows the newest ask. An open detail view stays put.
    fn follow(&mut self, idx: usize) {
        if !self.inspect {
            self.selected = Some(idx);
        }
    }
}

impl Default for Ask {
    fn default() -> Self {
        Self {
            id: String::new(),
            question: String::new(),
            brain_id: String::new(),
            as_of: String::new(),
            personal: 0,
            granted: 0,
            retrieve_k: 0,
            merged: 0,
            top_rrf: 0.0,
            admitted: 0,
            candidates: 0,
            conflicts: 0,
            top_noul: 0.0,
            threshold: 0.0,
            calibrated: false,
            empty: false,
            total_ms: 0,
            ranked: Vec::new(),
            complete: false,
            failed: false,
            kind: String::new(),
            kind_confidence: None,
            model_choice: String::new(),
            fell_back: None,
            chitchat: false,
            embed_dims: None,
            stages: Vec::new(),
            drafts: Vec::new(),
            generate_citations: None,
            verify_rounds: Vec::new(),
            regens: 0,
            published: None,
            skip_generate: false,
        }
    }
}

fn parse_ranked(v: Option<&Value>) -> Vec<CandidateRow> {
    let arr = json_array(v);
    arr.into_iter()
        .filter_map(|row| {
            let admitted = row.get("admitted")?.as_bool()?;
            let route = row
                .get("route")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| {
                    if admitted {
                        "include".into()
                    } else {
                        "exclude".into()
                    }
                });
            Some(CandidateRow {
                id: opt_str(&row, "id"),
                statement: row.get("statement")?.as_str()?.to_string(),
                text: opt_str(&row, "text"),
                noul: row.get("noul")?.as_f64()?,
                origin: row.get("origin")?.as_str().unwrap_or("?").to_string(),
                admitted,
                injection: opt_f64(&row, "injection"),
                contradicts: opt_f64(&row, "contradicts"),
                relevant: opt_f64(&row, "relevant"),
                evidence: opt_f64(&row, "evidence"),
                route,
                rrf: opt_f64(&row, "rrf"),
                source: opt_str(&row, "source"),
                grantor: opt_str(&row, "grantor"),
            })
        })
        .collect()
}

fn parse_verdicts(v: Option<&Value>) -> Vec<VerdictRow> {
    json_array(v)
        .into_iter()
        .filter_map(|row| {
            Some(VerdictRow {
                claim: row.get("claim")?.as_str()?.to_string(),
                verdict: row.get("verdict")?.as_str()?.to_string(),
                memory_index: row.get("memory_index").and_then(Value::as_u64),
            })
        })
        .collect()
}

fn json_array(v: Option<&Value>) -> Vec<Value> {
    match v {
        Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_default(),
        Some(Value::Array(a)) => a.clone(),
        _ => Vec::new(),
    }
}

fn opt_f64(row: &Value, key: &str) -> Option<f64> {
    row.get(key).and_then(Value::as_f64)
}

fn opt_str(row: &Value, key: &str) -> String {
    row.get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

enum LogSource {
    Stdin,
    FollowFile(String),
}

fn spawn_log_reader(tx: mpsc::Sender<String>, source: LogSource) {
    std::thread::spawn(move || match source {
        LogSource::Stdin => {
            use std::io::BufRead;
            for line in io::stdin().lock().lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        }
        LogSource::FollowFile(path) => {
            use std::fs::File;
            use std::io::{BufRead, BufReader};
            let Ok(file) = File::open(&path) else {
                eprintln!("recall-dash: cannot open {path}");
                return;
            };
            let mut reader = BufReader::new(file);
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => std::thread::sleep(Duration::from_millis(200)),
                    Ok(_) => {
                        if tx.send(line.trim_end_matches('\n').to_string()).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        }
    });
}

fn main() -> io::Result<()> {
    let source = match std::env::args().nth(1) {
        Some(path) => LogSource::FollowFile(path),
        None => LogSource::Stdin,
    };

    let (tx, rx) = mpsc::channel::<String>();
    spawn_log_reader(tx, source);

    enable_raw_mode().map_err(|e| {
        io::Error::new(
            io::ErrorKind::NotConnected,
            format!(
                "{e}. Run recall-dash from an interactive terminal. \
                 If dev.sh is already running, prefer: recall-dash /tmp/recall.jsonl"
            ),
        )
    })?;
    let mut out = io::stdout();
    execute!(out, EnterAlternateScreen)?;
    let mut term = Terminal::new(CrosstermBackend::new(out))?;

    let mut app = App {
        started: Some(Instant::now()),
        ..Default::default()
    };

    let res = loop {
        for line in rx.try_iter().take(500) {
            app.ingest(&line);
        }
        if let Err(e) = term.draw(|f| draw(f, &mut app)) {
            break Err(e);
        }
        if event::poll(Duration::from_millis(120))? {
            if let Event::Key(k) = event::read()? {
                let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
                match k.code {
                    KeyCode::Char('c') if ctrl => break Ok(()),
                    KeyCode::Char('q') => break Ok(()),
                    KeyCode::Esc => {
                        if app.inspect {
                            app.inspect = false;
                        } else {
                            break Ok(());
                        }
                    }
                    KeyCode::Enter => {
                        if app.selected.is_some() && !app.asks.is_empty() {
                            app.inspect = !app.inspect;
                            if app.inspect {
                                app.scroll = 0;
                            }
                        }
                    }
                    KeyCode::Up | KeyCode::Char('k') => {
                        if app.inspect {
                            app.scroll = app.scroll.saturating_sub(1);
                        } else {
                            app.move_selection(-1);
                        }
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        if app.inspect {
                            app.scroll = app.scroll.saturating_add(1);
                        } else {
                            app.move_selection(1);
                        }
                    }
                    KeyCode::PageUp | KeyCode::Char('u')
                        if app.inspect && (k.code != KeyCode::Char('u') || ctrl) =>
                    {
                        let page = app.view_rows.max(1) as u16;
                        app.scroll = app.scroll.saturating_sub(page);
                    }
                    KeyCode::PageDown | KeyCode::Char('d')
                        if app.inspect && (k.code != KeyCode::Char('d') || ctrl) =>
                    {
                        let page = app.view_rows.max(1) as u16;
                        app.scroll = app.scroll.saturating_add(page);
                    }
                    KeyCode::Home | KeyCode::Char('g') if app.inspect => app.scroll = 0,
                    KeyCode::End | KeyCode::Char('G') if app.inspect => app.scroll = u16::MAX,
                    _ => {}
                }
            }
        }
    };

    disable_raw_mode()?;
    execute!(term.backend_mut(), LeaveAlternateScreen)?;
    term.show_cursor()?;
    res
}

const FG: Color = Color::Rgb(205, 214, 244);
const DIM: Color = Color::Rgb(108, 112, 134);
const OK: Color = Color::Rgb(166, 227, 161);
const WARN: Color = Color::Rgb(249, 226, 175);
const BAD: Color = Color::Rgb(243, 139, 168);
const ACCENT: Color = Color::Rgb(137, 180, 250);
const MAUVE: Color = Color::Rgb(203, 166, 247);

fn block(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(DIM))
        .title(Span::styled(
            format!(" {title} "),
            Style::new().fg(MAUVE).add_modifier(Modifier::BOLD),
        ))
}

fn draw(f: &mut Frame, app: &mut App) {
    if app.inspect {
        draw_inspect(f, app);
        return;
    }
    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(14),
        Constraint::Length(7),
        Constraint::Min(8),
        Constraint::Length(6),
    ])
    .split(f.area());

    header(f, rows[0], app);

    let mid =
        Layout::horizontal([Constraint::Percentage(62), Constraint::Percentage(38)]).split(rows[1]);
    stages(f, mid[0], app);
    admission(f, mid[1], app);

    asks_table(f, rows[2], app);
    ask_detail(f, rows[3], app);
    events(f, rows[4], app);
}

fn header(f: &mut Frame, area: Rect, app: &App) {
    let up = app.started.map(|s| s.elapsed().as_secs()).unwrap_or(0);
    let rate = if up > 0 {
        app.total_asks as f64 * 60.0 / up as f64
    } else {
        0.0
    };
    let admit_pct = if app.total_asks > 0 {
        100.0 * (app.total_asks - app.empty_admits) as f64 / app.total_asks as f64
    } else {
        0.0
    };

    let sep = Span::styled("  │  ", Style::new().fg(DIM));
    let line = Line::from(vec![
        Span::styled(
            "  recall ",
            Style::new().fg(MAUVE).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("up {:02}:{:02}:{:02}", up / 3600, (up % 3600) / 60, up % 60),
            Style::new().fg(DIM),
        ),
        sep.clone(),
        Span::styled("asks ", Style::new().fg(DIM)),
        Span::styled(
            app.total_asks.to_string(),
            Style::new().fg(FG).add_modifier(Modifier::BOLD),
        ),
        sep.clone(),
        Span::styled(format!("{rate:.1}/min"), Style::new().fg(ACCENT)),
        sep.clone(),
        Span::styled("admitted ", Style::new().fg(DIM)),
        Span::styled(
            format!("{admit_pct:.0}%"),
            Style::new().fg(if admit_pct > 50.0 { OK } else { WARN }),
        ),
        sep.clone(),
        Span::styled("empty ", Style::new().fg(DIM)),
        Span::styled(app.empty_admits.to_string(), Style::new().fg(WARN)),
        sep.clone(),
        Span::styled("errors ", Style::new().fg(DIM)),
        Span::styled(
            app.errors.to_string(),
            Style::new().fg(if app.errors == 0 { OK } else { BAD }),
        ),
    ]);
    f.render_widget(Paragraph::new(line).block(block("service")), area);
}

fn stages(f: &mut Frame, area: Rect, app: &App) {
    let inner = block("stage latency");
    let region = inner.inner(area);
    f.render_widget(inner, area);

    let slots = Layout::vertical([Constraint::Length(2); 6]).split(region);
    for (i, name) in STAGES.iter().enumerate() {
        let Some(s) = app.stages.get(*name) else {
            continue;
        };
        let split =
            Layout::horizontal([Constraint::Length(30), Constraint::Min(10)]).split(slots[i]);

        let colour = match s.p50() {
            0..=200 => OK,
            201..=2000 => WARN,
            _ => BAD,
        };
        let label = Line::from(vec![
            Span::styled(format!(" {name:<9}"), Style::new().fg(FG)),
            Span::styled(
                format!("{:>6}ms", s.last()),
                Style::new().fg(colour).add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!(" p50 {:>5}", s.p50()), Style::new().fg(DIM)),
            Span::styled(format!(" max {:>5}", s.max()), Style::new().fg(DIM)),
        ]);
        f.render_widget(Paragraph::new(label), split[0]);

        let data: Vec<u64> = s.samples.iter().copied().collect();
        f.render_widget(
            Sparkline::default()
                .data(&data)
                .style(Style::new().fg(colour)),
            split[1],
        );
    }
}

fn admission(f: &mut Frame, area: Rect, app: &App) {
    let inner = block("admission");
    let region = inner.inner(area);
    f.render_widget(inner, area);

    let recent: Vec<&Ask> = app.asks.iter().rev().take(40).collect();
    let avg_cands = if recent.is_empty() {
        0.0
    } else {
        recent.iter().map(|a| a.candidates as f64).sum::<f64>() / recent.len() as f64
    };
    let avg_noul = if recent.is_empty() {
        0.0
    } else {
        recent.iter().map(|a| a.top_noul).sum::<f64>() / recent.len() as f64
    };

    let rows = Layout::vertical([Constraint::Length(1); 6]).split(region);
    let kv = |k: &str, v: String, c: Color| {
        Line::from(vec![
            Span::styled(format!(" {k:<14}"), Style::new().fg(DIM)),
            Span::styled(v, Style::new().fg(c).add_modifier(Modifier::BOLD)),
        ])
    };

    f.render_widget(
        Paragraph::new(kv("threshold", format!("{:.3}", app.threshold), ACCENT)),
        rows[0],
    );
    f.render_widget(
        Paragraph::new(kv(
            "calibrated",
            if app.calibrated {
                "yes".into()
            } else {
                "NO (provisional)".to_string()
            },
            if app.calibrated { OK } else { WARN },
        )),
        rows[1],
    );
    f.render_widget(
        Paragraph::new(kv("avg top noul", format!("{avg_noul:.3}"), FG)),
        rows[2],
    );
    f.render_widget(
        Paragraph::new(kv("avg candidates", format!("{avg_cands:.1}"), FG)),
        rows[3],
    );

    // Admitted vs empty over the recent window.
    let empties = recent.iter().filter(|a| a.empty).count();
    let ratio = if recent.is_empty() {
        0.0
    } else {
        1.0 - empties as f64 / recent.len() as f64
    };
    f.render_widget(
        Gauge::default()
            .gauge_style(Style::new().fg(if ratio > 0.5 { OK } else { WARN }))
            .ratio(ratio)
            .label(format!("{:.0}% answered", ratio * 100.0)),
        rows[5],
    );
}

fn asks_table(f: &mut Frame, area: Rect, app: &App) {
    let sel = app
        .selected
        .unwrap_or_else(|| app.asks.len().saturating_sub(1));
    let rows: Vec<Row> = app
        .asks
        .iter()
        .enumerate()
        .rev()
        .take(area.height.saturating_sub(3) as usize)
        .map(|(i, a)| {
            let (verdict, c) = row_verdict(a);
            let q = trunc_display(&a.question, 48);
            let style = if Some(i) == app.selected {
                Style::new().fg(FG).add_modifier(Modifier::REVERSED)
            } else {
                Style::new().fg(FG)
            };
            Row::new(vec![
                Cell::from(a.id.chars().take(8).collect::<String>()).style(style),
                Cell::from(q).style(style),
                Cell::from(format!("{}/{}", a.admitted, a.candidates)).style(style),
                Cell::from(format!("{:.3}", a.top_noul)).style(style),
                Cell::from(verdict).style(Style::new().fg(c).add_modifier(Modifier::BOLD)),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(10),
            Constraint::Min(20),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(10),
        ],
    )
    .header(
        Row::new(vec!["ask", "question", "adm/cand", "top noul", "verdict"])
            .style(Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)),
    );
    let title = format!("recent asks   j/k select   enter opens   sel={sel}");
    let table = table.block(block(&title));

    f.render_widget(table, area);
}

fn ask_detail(f: &mut Frame, area: Rect, app: &App) {
    let inner = block("ask detail   enter for the pipeline");
    let region = inner.inner(area);
    f.render_widget(inner, area);

    let Some(idx) = app.selected.or_else(|| app.asks.len().checked_sub(1)) else {
        f.render_widget(
            Paragraph::new("  no asks yet").style(Style::new().fg(DIM)),
            region,
        );
        return;
    };
    let Some(ask) = app.asks.get(idx) else {
        return;
    };

    let meta = Line::from(vec![
        Span::styled(" brain ", Style::new().fg(DIM)),
        Span::styled(trunc_display(&ask.brain_id, 36), Style::new().fg(FG)),
        Span::styled("  │  retrieve ", Style::new().fg(DIM)),
        Span::styled(
            format!("{} personal + {} granted", ask.personal, ask.granted),
            Style::new().fg(ACCENT),
        ),
        Span::styled("  │  ", Style::new().fg(DIM)),
        Span::styled(format!("{}ms", ask.total_ms), Style::new().fg(DIM)),
    ]);
    let question = Paragraph::new(Line::from(vec![
        Span::styled(" Q  ", Style::new().fg(MAUVE).add_modifier(Modifier::BOLD)),
        Span::styled(ask.question.clone(), Style::new().fg(FG)),
    ]));

    let list_h = region.height.saturating_sub(4) as usize;
    let cand_rows: Vec<Row> = ask
        .ranked
        .iter()
        .take(list_h)
        .map(|c| {
            let mark = if c.admitted { "✓" } else { "·" };
            let colour = if c.admitted {
                OK
            } else if c.noul >= app.threshold * 0.8 {
                WARN
            } else {
                DIM
            };
            Row::new(vec![
                Cell::from(mark).style(Style::new().fg(colour).add_modifier(Modifier::BOLD)),
                Cell::from(format!("{:.3}", c.noul)).style(Style::new().fg(colour)),
                Cell::from(c.origin.clone()).style(Style::new().fg(DIM)),
                Cell::from(trunc_display(&c.statement, 72)).style(Style::new().fg(FG)),
            ])
        })
        .collect();

    let slots = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(2),
        Constraint::Min(3),
    ])
    .split(region);
    f.render_widget(meta, slots[0]);
    f.render_widget(question, slots[1]);

    let table = Table::new(
        cand_rows,
        [
            Constraint::Length(2),
            Constraint::Length(7),
            Constraint::Length(10),
            Constraint::Min(20),
        ],
    )
    .header(
        Row::new(vec!["", "noul", "origin", "candidate (ranked by noul)"])
            .style(Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)),
    );
    f.render_widget(table, slots[2]);
}

fn trunc_display(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max).collect::<String>())
    }
}

fn row_verdict(a: &Ask) -> (&'static str, Color) {
    let late_fail = a.complete
        && a.published.is_none()
        && !a.skip_generate
        && a.admitted > 0
        && a.stages.iter().any(|s| !s.ok);
    if (!a.complete && a.failed) || late_fail {
        ("failed", BAD)
    } else if !a.complete {
        ("running", DIM)
    } else if a.empty {
        ("empty", WARN)
    } else {
        ("answered", OK)
    }
}

fn draw_inspect(f: &mut Frame, app: &mut App) {
    let area = f.area();
    let idx = app.selected.or_else(|| app.asks.len().checked_sub(1));
    let Some(idx) = idx.filter(|i| app.asks.get(*i).is_some()) else {
        f.render_widget(
            Paragraph::new("  no ask selected").block(block("request")),
            area,
        );
        return;
    };

    let width = area.width.saturating_sub(4) as usize;
    let head = inspect_head(&app.asks[idx]);
    let body = detail_lines(&app.asks[idx], width.max(24));

    let rows = Layout::vertical([Constraint::Length(5), Constraint::Min(1)]).split(area);
    f.render_widget(
        Paragraph::new(head).block(block("request   esc back")),
        rows[0],
    );

    let view_guess = rows[1].height.saturating_sub(2) as usize;
    let max = body.len().saturating_sub(view_guess.max(1));
    if app.scroll as usize > max {
        app.scroll = max as u16;
    }
    let title_pos = if body.is_empty() {
        0
    } else {
        (app.scroll as usize + 1).min(body.len())
    };
    let title = format!(
        "pipeline   {title_pos}/{}   j/k scroll   esc back",
        body.len()
    );
    let titled = block(&title);
    let region = titled.inner(rows[1]);
    app.view_rows = (region.height as usize).max(1);
    f.render_widget(titled, rows[1]);
    f.render_widget(Paragraph::new(body).scroll((app.scroll, 0)), region);
}

fn inspect_head(ask: &Ask) -> Vec<Line<'static>> {
    let (verdict, color) = row_verdict(ask);
    let when = if ask.as_of.is_empty() || ask.as_of == "None" {
        "now".to_string()
    } else {
        ask.as_of.clone()
    };
    vec![
        Line::from(vec![
            Span::styled(
                format!("  {}", ask.id),
                Style::new().fg(FG).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("   {verdict}"),
                Style::new().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("   {}ms", ask.total_ms), Style::new().fg(DIM)),
        ]),
        Line::from(vec![
            Span::styled("  brain ", Style::new().fg(DIM)),
            Span::styled(ask.brain_id.clone(), Style::new().fg(FG)),
            Span::styled("   as of ", Style::new().fg(DIM)),
            Span::styled(when, Style::new().fg(FG)),
        ]),
        Line::from(vec![
            Span::styled("  Q ", Style::new().fg(MAUVE).add_modifier(Modifier::BOLD)),
            Span::styled(ask.question.clone(), Style::new().fg(FG)),
        ]),
    ]
}

fn detail_lines(ask: &Ask, width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    push_wrapped(
        &mut lines,
        "  Q ",
        "    ",
        &ask.question,
        width,
        Style::new().fg(FG),
    );
    lines.push(Line::from(""));

    push_kind(&mut lines, ask);
    lines.push(Line::from(""));
    push_embed(&mut lines, ask);
    lines.push(Line::from(""));
    push_retrieve(&mut lines, ask);
    lines.push(Line::from(""));
    push_admit(&mut lines, ask, width);
    lines.push(Line::from(""));
    push_generate(&mut lines, ask, width);
    lines.push(Line::from(""));
    push_verify(&mut lines, ask, width);
    lines.push(Line::from(""));
    push_published(&mut lines, ask, width);
    lines
}

fn push_kind(lines: &mut Vec<Line<'static>>, ask: &Ask) {
    push_stage(lines, "kind", &stage_hits(ask, "kind"), None);
    if ask.kind.is_empty() && ask.kind_confidence.is_none() {
        return;
    }
    kv(lines, "chosen", or_dash(&ask.kind), ACCENT);
    kv(
        lines,
        "confidence",
        &ask.kind_confidence
            .map(|n| format!("{n:.3}"))
            .unwrap_or_else(|| "—".into()),
        FG,
    );
    kv(lines, "model choice", or_dash(&ask.model_choice), FG);
    let fell = match ask.fell_back {
        Some(true) => "yes — confidence below minimum, used atomic_lookup",
        Some(false) => "no",
        None => "—",
    };
    kv(
        lines,
        "fell back",
        fell,
        if ask.fell_back == Some(true) {
            WARN
        } else {
            FG
        },
    );
    if ask.chitchat {
        kv(
            lines,
            "branch",
            "chitchat — later memory stages do not run",
            MAUVE,
        );
    }
}

fn push_embed(lines: &mut Vec<Line<'static>>, ask: &Ask) {
    push_stage(
        lines,
        "embed",
        &stage_hits(ask, "embed"),
        skip_reason(ask, "embed"),
    );
    if let Some(dims) = ask.embed_dims {
        kv(lines, "dimensions", &dims.to_string(), FG);
    }
}

fn push_retrieve(lines: &mut Vec<Line<'static>>, ask: &Ask) {
    push_stage(
        lines,
        "retrieve",
        &stage_hits(ask, "retrieve"),
        skip_reason(ask, "retrieve"),
    );
    if ask.personal == 0 && ask.granted == 0 && ask.merged == 0 && ask.retrieve_k == 0 {
        return;
    }
    kv(lines, "personal", &ask.personal.to_string(), FG);
    kv(lines, "granted", &ask.granted.to_string(), FG);
    kv(lines, "k", &ask.retrieve_k.to_string(), FG);
    kv(lines, "merged", &ask.merged.to_string(), ACCENT);
    kv(lines, "top rrf", &format!("{:.4}", ask.top_rrf), FG);
}

fn push_admit(lines: &mut Vec<Line<'static>>, ask: &Ask, width: usize) {
    push_stage(
        lines,
        "admit",
        &stage_hits(ask, "admit"),
        skip_reason(ask, "admit"),
    );
    if !ask.complete && ask.ranked.is_empty() && ask.candidates == 0 {
        return;
    }
    kv(lines, "candidates", &ask.candidates.to_string(), FG);
    kv(lines, "included", &ask.admitted.to_string(), OK);
    kv(lines, "conflicts", &ask.conflicts.to_string(), WARN);
    let cal = if ask.calibrated {
        "calibrated"
    } else {
        "provisional"
    };
    kv(
        lines,
        "threshold",
        &format!("{:.3}  {cal}", ask.threshold),
        ACCENT,
    );
    lines.push(Line::from(Span::styled("     route", Style::new().fg(DIM))));
    push_wrapped(
        lines,
        "       ",
        "       ",
        "injection high, relevant low, or evidence under the threshold excludes; a contradiction is a conflict",
        width,
        Style::new().fg(DIM),
    );
    if ask.ranked.is_empty() {
        kv(lines, "scores", "not in the log yet", DIM);
        return;
    }
    for c in &ask.ranked {
        push_candidate(lines, c, width);
    }
}

fn push_generate(lines: &mut Vec<Line<'static>>, ask: &Ask, width: usize) {
    push_stage(
        lines,
        "generate",
        &stage_hits(ask, "generate"),
        skip_reason(ask, "generate"),
    );
    if let Some(n) = ask.generate_citations {
        kv(lines, "citations", &n.to_string(), FG);
    }
    if ask.chitchat {
        kv(lines, "mode", "chitchat, no citations", MAUVE);
    }
    for (i, draft) in ask.drafts.iter().enumerate() {
        let label = if ask.drafts.len() > 1 {
            format!("draft {}", i + 1)
        } else {
            "draft".into()
        };
        lines.push(Line::from(Span::styled(
            format!("     {label}"),
            Style::new().fg(DIM),
        )));
        push_wrapped(
            lines,
            "       ",
            "       ",
            draft,
            width,
            Style::new().fg(FG),
        );
    }
}

fn push_verify(lines: &mut Vec<Line<'static>>, ask: &Ask, width: usize) {
    push_stage(
        lines,
        "verify",
        &stage_hits(ask, "verify"),
        skip_reason(ask, "verify"),
    );
    if ask.regens > 0 {
        kv(
            lines,
            "regenerated",
            &format!("{} time(s) after a failed verify", ask.regens),
            WARN,
        );
    }
    for round in &ask.verify_rounds {
        kv(lines, "attempt", &round.attempt.to_string(), FG);
        kv(
            lines,
            "claims",
            &format!("{}/{} support", round.supported, round.claims),
            if round.all_ok { OK } else { WARN },
        );
        kv(
            lines,
            "all support",
            if round.all_ok { "yes" } else { "no" },
            if round.all_ok { OK } else { WARN },
        );
        for v in &round.verdicts {
            let color = match v.verdict.as_str() {
                "supports" => OK,
                "fabricated" => BAD,
                _ => WARN,
            };
            let mem = v
                .memory_index
                .map(|n| format!("[memory_{n}] "))
                .unwrap_or_default();
            lines.push(Line::from(vec![
                Span::styled(
                    format!("     {:<12}", v.verdict),
                    Style::new().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(mem, Style::new().fg(DIM)),
            ]));
            push_wrapped(
                lines,
                "       ",
                "       ",
                &v.claim,
                width,
                Style::new().fg(FG),
            );
        }
    }
}

fn push_published(lines: &mut Vec<Line<'static>>, ask: &Ask, width: usize) {
    lines.push(Line::from(Span::styled(
        "  published",
        Style::new().fg(MAUVE).add_modifier(Modifier::BOLD),
    )));
    match &ask.published {
        Some(text) => {
            kv(
                lines,
                "empty",
                if ask.empty { "yes" } else { "no" },
                if ask.empty { WARN } else { OK },
            );
            push_wrapped(lines, "     ", "     ", text, width, Style::new().fg(FG));
        }
        None => kv(lines, "text", "not published yet", DIM),
    }
}

fn stage_hits<'a>(ask: &'a Ask, name: &str) -> Vec<&'a StageHit> {
    ask.stages.iter().filter(|s| s.name == name).collect()
}

fn skip_reason(ask: &Ask, name: &str) -> Option<&'static str> {
    if ask.chitchat && matches!(name, "embed" | "retrieve" | "admit" | "verify") {
        return Some("skipped   chitchat");
    }
    if ask.skip_generate && matches!(name, "generate" | "verify") {
        return Some("skipped   nothing admitted");
    }
    if ask.complete {
        return Some("not run");
    }
    None
}

fn push_stage(
    lines: &mut Vec<Line<'static>>,
    name: &str,
    hits: &[&StageHit],
    skipped: Option<&str>,
) {
    if hits.is_empty() {
        let status = skipped.unwrap_or("waiting");
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {name:<10}"),
                Style::new().fg(FG).add_modifier(Modifier::BOLD),
            ),
            Span::styled(status.to_string(), Style::new().fg(DIM)),
        ]));
        return;
    }
    for (i, hit) in hits.iter().enumerate() {
        let mut spans = vec![
            Span::styled(
                format!("  {name:<10}"),
                Style::new().fg(FG).add_modifier(Modifier::BOLD),
            ),
            if hit.ok {
                Span::styled(
                    format!("{:>6}ms  ok", hit.ms),
                    Style::new().fg(OK).add_modifier(Modifier::BOLD),
                )
            } else {
                Span::styled(
                    format!("{:>6}ms  FAILED", hit.ms),
                    Style::new().fg(BAD).add_modifier(Modifier::BOLD),
                )
            },
        ];
        if hits.len() > 1 {
            spans.push(Span::styled(
                format!("   run {}", i + 1),
                Style::new().fg(DIM),
            ));
        }
        lines.push(Line::from(spans));
        if !hit.ok && !hit.error.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("     {}", hit.error),
                Style::new().fg(BAD),
            )));
        }
    }
}

fn push_candidate(lines: &mut Vec<Line<'static>>, c: &CandidateRow, width: usize) {
    let ev = c.evidence.unwrap_or(c.noul);
    let color = match c.route.as_str() {
        "include" => OK,
        "conflict" => WARN,
        _ => DIM,
    };
    let scores = format!(
        "{:<9} ev {ev:.3}  rel {}  inj {}  con {}  rrf {}  {}",
        c.route,
        fmt_score(c.relevant),
        fmt_score(c.injection),
        fmt_score(c.contradicts),
        fmt_score(c.rrf),
        c.origin,
    );
    lines.push(Line::from(Span::styled(
        format!("     {scores}"),
        Style::new().fg(color),
    )));
    push_wrapped(
        lines,
        "       ",
        "       ",
        &c.statement,
        width,
        Style::new().fg(FG),
    );
    if !c.text.is_empty() {
        push_wrapped(
            lines,
            "       text ",
            "            ",
            &c.text,
            width,
            Style::new().fg(DIM),
        );
    }
    let mut meta = String::new();
    if !c.id.is_empty() {
        meta.push_str(&c.id.chars().take(8).collect::<String>());
    }
    if !c.grantor.is_empty() {
        if !meta.is_empty() {
            meta.push_str("  ");
        }
        meta.push_str("grantor ");
        meta.push_str(&c.grantor);
    }
    if !c.source.is_empty() {
        if !meta.is_empty() {
            meta.push_str("  ");
        }
        meta.push_str("source ");
        meta.push_str(&c.source);
    }
    if !meta.is_empty() {
        lines.push(Line::from(Span::styled(
            format!("       {meta}"),
            Style::new().fg(DIM),
        )));
    }
}

fn fmt_score(v: Option<f64>) -> String {
    v.map(|n| format!("{n:.3}")).unwrap_or_else(|| "—".into())
}

fn or_dash(s: &str) -> &str {
    if s.is_empty() {
        "—"
    } else {
        s
    }
}

fn tidy_debug_opt(s: &str) -> String {
    let s = s.trim();
    if s.is_empty() || s == "None" {
        return String::new();
    }
    s.strip_prefix("Some(")
        .and_then(|rest| rest.strip_suffix(')'))
        .unwrap_or(s)
        .to_string()
}

fn kv(lines: &mut Vec<Line<'static>>, key: &str, val: &str, color: Color) {
    lines.push(Line::from(vec![
        Span::styled(format!("     {key:<14}"), Style::new().fg(DIM)),
        Span::styled(val.to_string(), Style::new().fg(color)),
    ]));
}

fn push_wrapped(
    out: &mut Vec<Line<'static>>,
    prefix: &str,
    cont: &str,
    text: &str,
    width: usize,
    style: Style,
) {
    let width = width.max(24);
    if text.is_empty() {
        out.push(Line::from(Span::styled(
            prefix.to_string(),
            Style::new().fg(DIM),
        )));
        return;
    }
    let mut rest = text;
    let mut first = true;
    while !rest.is_empty() {
        let p = if first { prefix } else { cont };
        let budget = width.saturating_sub(p.chars().count()).max(8);
        let bytes = wrap_end(rest, budget);
        if bytes == 0 {
            break;
        }
        let (head, tail) = rest.split_at(bytes.min(rest.len()));
        let tail = tail.trim_start();
        out.push(Line::from(vec![
            Span::styled(p.to_string(), Style::new().fg(DIM)),
            Span::styled(head.trim_end().to_string(), style),
        ]));
        if tail.is_empty() {
            break;
        }
        rest = tail;
        first = false;
    }
}

fn wrap_end(s: &str, budget: usize) -> usize {
    if s.chars().count() <= budget {
        return s.len();
    }
    let mut end = 0;
    let mut last_space = 0;
    let mut chars = 0;
    for (i, c) in s.char_indices() {
        if chars == budget {
            break;
        }
        end = i + c.len_utf8();
        chars += 1;
        if c.is_whitespace() {
            last_space = end;
        }
    }
    if last_space > 0 && s[..last_space].chars().count() >= budget / 2 {
        last_space
    } else {
        end
    }
}

fn events(f: &mut Frame, area: Rect, app: &App) {
    let items: Vec<ListItem> = app
        .events
        .iter()
        .rev()
        .take(area.height.saturating_sub(2) as usize)
        .map(|(kind, detail)| {
            let c = match kind.as_str() {
                k if k.contains("FAILED") => BAD,
                "REGEN" | "RETRY" => WARN,
                "WITHHELD" => MAUVE,
                _ => DIM,
            };
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!(" {kind:<16}"),
                    Style::new().fg(c).add_modifier(Modifier::BOLD),
                ),
                Span::styled(detail.clone(), Style::new().fg(FG)),
            ]))
        })
        .collect();

    f.render_widget(List::new(items).block(block("events")), area);
}

#[cfg(test)]
mod tests {
    use super::App;

    #[test]
    fn ingests_stage_and_admission_lines() {
        let mut app = App::default();
        app.ingest(
            r#"{"message":"ask started","ask_id":"abc-def","brain_id":"brain-1","question":"When is Ana's birthday?"}"#,
        );
        app.ingest(
            r#"{"message":"retrieved from nexus","ask_id":"abc-def","personal":20,"granted":12}"#,
        );
        app.ingest(r#"{"message":"stage ok","stage":"embed","ms":42,"ask_id":"abc-def"}"#);
        app.ingest(r#"{"message":"stage failed","stage":"admit","ms":900,"ask_id":"abc-def","error":"boom"}"#);
        app.ingest(
            r#"{"message":"admission complete","ask_id":"abc-def","question":"When is Ana's birthday?","candidates":32,"admitted":0,"threshold":0.15,"top_noul":0.04,"ranked":"[{\"statement\":\"Ana born March 14\",\"noul\":0.04,\"origin\":\"personal\",\"admitted\":false}]"}"#,
        );

        assert_eq!(app.stages["embed"].last(), 42);
        assert_eq!(app.stages["admit"].fails, 1);
        assert_eq!(app.errors, 1);
        assert_eq!(app.total_asks, 1);
        assert_eq!(app.empty_admits, 1, "admitted=0 is an empty admit");
        assert_eq!(app.asks[0].question, "When is Ana's birthday?");
        assert_eq!(app.asks[0].personal, 20);
        assert_eq!(app.asks[0].granted, 12);
        assert_eq!(app.asks[0].ranked.len(), 1);
        // 42 + 900 accumulated before the ask completed.
        assert_eq!(app.asks[0].total_ms, 942);
        assert!(app.ingest_noop_on_garbage());
    }

    #[test]
    fn inspect_lists_each_stage_choice() {
        let mut app = App::default();
        app.ingest(
            r#"{"message":"ask started","ask_id":"abc","brain_id":"brain-1","question":"When is Ana's birthday?","as_of":"Some(2024-03-01T00:00:00Z)"}"#,
        );
        app.ingest(
            r#"{"message":"kind chosen","ask_id":"abc","kind":"atomic_lookup","confidence":0.42,"model_choice":"temporal","fell_back":true,"chitchat":false}"#,
        );
        app.ingest(r#"{"message":"stage ok","stage":"kind","ms":11,"ask_id":"abc"}"#);
        app.ingest(r#"{"message":"embedded","ask_id":"abc","dims":1024}"#);
        app.ingest(r#"{"message":"stage ok","stage":"embed","ms":20,"ask_id":"abc"}"#);
        app.ingest(
            r#"{"message":"retrieved from nexus","ask_id":"abc","personal":4,"granted":1,"k":64}"#,
        );
        app.ingest(r#"{"message":"retrieve merged","ask_id":"abc","merged":5,"top_rrf":0.031}"#);
        app.ingest(r#"{"message":"stage ok","stage":"retrieve","ms":30,"ask_id":"abc"}"#);
        app.ingest(
            r#"{"message":"admission complete","ask_id":"abc","candidates":5,"admitted":1,"conflicts":1,"threshold":0.55,"calibrated":true,"top_noul":0.9,"ranked":"[{\"id\":\"11111111-1111-1111-1111-111111111111\",\"statement\":\"Ana was born on March 14\",\"text\":\"chunk about Ana\",\"noul\":0.9,\"injection\":0.01,\"contradicts\":0.02,\"relevant\":0.88,\"evidence\":0.9,\"rrf\":0.016,\"origin\":\"personal\",\"source\":\"notes\",\"grantor\":null,\"route\":\"include\",\"admitted\":true}]"}"#,
        );
        app.ingest(r#"{"message":"stage ok","stage":"admit","ms":40,"ask_id":"abc"}"#);
        app.ingest(
            r#"{"message":"generated","ask_id":"abc","chitchat":false,"citations":1,"answer":"Ana was born on \"March 14\" [memory_0]."}"#,
        );
        app.ingest(r#"{"message":"stage ok","stage":"generate","ms":50,"ask_id":"abc"}"#);
        app.ingest(
            r#"{"message":"verify verdicts","ask_id":"abc","claims":1,"supported":1,"all_ok":true,"attempt":1,"verdicts":"[{\"claim\":\"Ana was born on \\\"March 14\\\" [memory_0].\",\"verdict\":\"supports\",\"memory_index\":0}]"}"#,
        );
        app.ingest(r#"{"message":"stage ok","stage":"verify","ms":15,"ask_id":"abc"}"#);
        app.ingest(
            r#"{"message":"answer published","ask_id":"abc","chitchat":false,"empty":false,"answer":"Ana was born on \"March 14\" [memory_0]."}"#,
        );

        let ask = &app.asks[0];
        assert_eq!(ask.kind, "atomic_lookup");
        assert_eq!(ask.model_choice, "temporal");
        assert_eq!(ask.fell_back, Some(true));
        assert_eq!(ask.embed_dims, Some(1024));
        assert_eq!(ask.retrieve_k, 64);
        assert_eq!(ask.merged, 5);
        assert_eq!(ask.ranked[0].route, "include");
        assert_eq!(ask.ranked[0].source, "notes");
        assert_eq!(ask.drafts.len(), 1);
        assert_eq!(ask.verify_rounds[0].verdicts[0].verdict, "supports");
        assert_eq!(ask.verify_rounds[0].verdicts[0].memory_index, Some(0));
        assert!(ask.published.as_deref().unwrap().contains("March 14"));
        // Admission already counted the ask; publishing must not count it again.
        assert_eq!(app.total_asks, 1);

        app.inspect = true;
        let text = frame_text(&mut app, 110, 72);
        for needle in [
            "atomic_lookup",
            "temporal",
            "fell back",
            "1024",
            "include",
            "March 14",
            "supports",
            "published",
            "notes",
        ] {
            assert!(
                text.contains(needle),
                "detail view missing {needle}\n{text}"
            );
        }
    }

    fn frame_text(app: &mut App, w: u16, h: u16) -> String {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
        t.draw(|f| super::draw(f, app)).unwrap();
        let buf = t.backend().buffer().clone();
        let mut out = String::new();
        for y in 0..h {
            for x in 0..w {
                out.push_str(buf.cell((x, y)).unwrap().symbol());
            }
            out.push('\n');
        }
        out
    }

    /// Draw at an absurdly small size: layout slots and the recent-asks/events
    /// windows all index into areas derived from the terminal height, so a short
    /// terminal is where an off-by-one would panic.
    #[test]
    fn renders_without_panicking_when_cramped() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut app = App::default();
        app.ingest(r#"{"message":"stage ok","stage":"embed","ms":7,"ask_id":"a1"}"#);
        app.ingest(
            r#"{"message":"admission complete","ask_id":"a1","candidates":3,"admitted":1,"threshold":0.15,"top_noul":0.9}"#,
        );

        for (w, h) in [(20u16, 6u16), (80, 24), (200, 60)] {
            let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
            t.draw(|f| super::draw(f, &mut app)).unwrap();
        }
        app.inspect = true;
        for (w, h) in [(20u16, 6u16), (80, 24), (200, 60)] {
            let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
            t.draw(|f| super::draw(f, &mut app)).unwrap();
        }
    }
}

#[cfg(test)]
impl App {
    fn ingest_noop_on_garbage(&mut self) -> bool {
        let before = self.total_asks;
        self.ingest("not json at all");
        self.ingest(r#"{"message":"unrelated"}"#);
        self.total_asks == before
    }
}
