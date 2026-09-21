//! recall-dash — live TUI over the service's JSON log stream.
//!
//! Reads what the service already emits instead of adding an endpoint or a metrics
//! channel: every number here comes from existing `tracing` events.
//!
//!   LOG_FORMAT=json ./recall 2>&1 | recall-dash
//!   tail -f /tmp/recall.log | recall-dash
//!
//! q / Esc / Ctrl-C to quit.

use std::collections::{HashMap, VecDeque};
use std::io;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::execute;
use ratatui::prelude::*;
use ratatui::widgets::*;
use serde_json::Value;

const STAGES: [&str; 4] = ["embed", "retrieve", "admit", "generate"];
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

struct Ask {
    id: String,
    admitted: u64,
    candidates: u64,
    top_noul: f64,
    empty: bool,
    total_ms: u64,
}

#[derive(Default)]
struct App {
    stages: HashMap<String, Stage>,
    asks: VecDeque<Ask>,
    events: VecDeque<(String, String)>,
    inflight: HashMap<String, u64>,
    total_asks: u64,
    empty_admits: u64,
    errors: u64,
    threshold: f64,
    calibrated: bool,
    started: Option<Instant>,
}

impl App {
    fn stage(&mut self, name: &str) -> &mut Stage {
        self.stages.entry(name.to_string()).or_default()
    }

    fn ingest(&mut self, line: &str) {
        let Ok(v) = serde_json::from_str::<Value>(line) else { return };
        let msg = v.get("message").and_then(Value::as_str).unwrap_or("");
        let num = |k: &str| v.get(k).and_then(Value::as_f64);
        let text = |k: &str| v.get(k).and_then(Value::as_str).map(str::to_string);

        match msg {
            "stage ok" | "stage failed" => {
                let (Some(stage), Some(ms)) = (text("stage"), num("ms")) else { return };
                let ok = msg == "stage ok";
                self.stage(&stage).push(ms as u64, ok);
                if let Some(id) = text("ask_id") {
                    *self.inflight.entry(id).or_insert(0) += ms as u64;
                }
                if !ok {
                    self.errors += 1;
                    self.events.push_back((
                        format!("{stage} FAILED"),
                        text("error").unwrap_or_default(),
                    ));
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
                let total_ms = self.inflight.remove(&id).unwrap_or(0);
                self.total_asks += 1;
                if admitted == 0 {
                    self.empty_admits += 1;
                }
                self.asks.push_back(Ask {
                    id: id.chars().take(8).collect(),
                    admitted,
                    candidates: num("candidates").unwrap_or(0.0) as u64,
                    top_noul: num("top_noul").unwrap_or(0.0),
                    empty: admitted == 0,
                    total_ms,
                });
                if self.asks.len() > 200 {
                    self.asks.pop_front();
                }
            }
            "withheld secret-sensitivity rows" => self
                .events
                .push_back(("WITHHELD".into(), format!("{} secret rows", num("withheld").unwrap_or(0.0)))),
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
}

fn main() -> io::Result<()> {
    let (tx, rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        use std::io::BufRead;
        for line in io::stdin().lock().lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });

    enable_raw_mode()?;
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
        if let Err(e) = term.draw(|f| draw(f, &app)) {
            break Err(e);
        }
        if event::poll(Duration::from_millis(120))? {
            if let Event::Key(k) = event::read()? {
                let quit = matches!(k.code, KeyCode::Char('q') | KeyCode::Esc)
                    || (k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL));
                if quit {
                    break Ok(());
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

fn draw(f: &mut Frame, app: &App) {
    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(10),
        Constraint::Min(6),
        Constraint::Length(8),
    ])
    .split(f.area());

    header(f, rows[0], app);

    let mid = Layout::horizontal([Constraint::Percentage(62), Constraint::Percentage(38)]).split(rows[1]);
    stages(f, mid[0], app);
    admission(f, mid[1], app);

    asks_table(f, rows[2], app);
    events(f, rows[3], app);
}

fn header(f: &mut Frame, area: Rect, app: &App) {
    let up = app.started.map(|s| s.elapsed().as_secs()).unwrap_or(0);
    let rate = if up > 0 { app.total_asks as f64 * 60.0 / up as f64 } else { 0.0 };
    let admit_pct = if app.total_asks > 0 {
        100.0 * (app.total_asks - app.empty_admits) as f64 / app.total_asks as f64
    } else {
        0.0
    };

    let sep = Span::styled("  │  ", Style::new().fg(DIM));
    let line = Line::from(vec![
        Span::styled("  recall ", Style::new().fg(MAUVE).add_modifier(Modifier::BOLD)),
        Span::styled(format!("up {:02}:{:02}:{:02}", up / 3600, (up % 3600) / 60, up % 60), Style::new().fg(DIM)),
        sep.clone(),
        Span::styled("asks ", Style::new().fg(DIM)),
        Span::styled(app.total_asks.to_string(), Style::new().fg(FG).add_modifier(Modifier::BOLD)),
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

    let slots = Layout::vertical([Constraint::Length(2); 4]).split(region);
    for (i, name) in STAGES.iter().enumerate() {
        let Some(s) = app.stages.get(*name) else { continue };
        let split = Layout::horizontal([Constraint::Length(30), Constraint::Min(10)]).split(slots[i]);

        let colour = match s.p50() {
            0..=200 => OK,
            201..=2000 => WARN,
            _ => BAD,
        };
        let label = Line::from(vec![
            Span::styled(format!(" {name:<9}"), Style::new().fg(FG)),
            Span::styled(format!("{:>6}ms", s.last()), Style::new().fg(colour).add_modifier(Modifier::BOLD)),
            Span::styled(format!(" p50 {:>5}", s.p50()), Style::new().fg(DIM)),
            Span::styled(format!(" max {:>5}", s.max()), Style::new().fg(DIM)),
        ]);
        f.render_widget(Paragraph::new(label), split[0]);

        let data: Vec<u64> = s.samples.iter().copied().collect();
        f.render_widget(
            Sparkline::default().data(&data).style(Style::new().fg(colour)),
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

    f.render_widget(Paragraph::new(kv("threshold", format!("{:.3}", app.threshold), ACCENT)), rows[0]);
    f.render_widget(
        Paragraph::new(kv(
            "calibrated",
            if app.calibrated { "yes".into() } else { "NO (provisional)".to_string() },
            if app.calibrated { OK } else { WARN },
        )),
        rows[1],
    );
    f.render_widget(Paragraph::new(kv("avg top noul", format!("{avg_noul:.3}"), FG)), rows[2]);
    f.render_widget(Paragraph::new(kv("avg candidates", format!("{avg_cands:.1}"), FG)), rows[3]);

    // Admitted vs empty over the recent window.
    let empties = recent.iter().filter(|a| a.empty).count();
    let ratio = if recent.is_empty() { 0.0 } else { 1.0 - empties as f64 / recent.len() as f64 };
    f.render_widget(
        Gauge::default()
            .gauge_style(Style::new().fg(if ratio > 0.5 { OK } else { WARN }))
            .ratio(ratio)
            .label(format!("{:.0}% answered", ratio * 100.0)),
        rows[5],
    );
}

fn asks_table(f: &mut Frame, area: Rect, app: &App) {
    let rows: Vec<Row> = app
        .asks
        .iter()
        .rev()
        .take(area.height.saturating_sub(3) as usize)
        .map(|a| {
            let (verdict, c) = if a.empty {
                ("empty", WARN)
            } else {
                ("answered", OK)
            };
            Row::new(vec![
                Cell::from(a.id.clone()).style(Style::new().fg(DIM)),
                Cell::from(format!("{}/{}", a.admitted, a.candidates)),
                Cell::from(format!("{:.3}", a.top_noul)).style(Style::new().fg(
                    if a.top_noul >= app.threshold { OK } else { BAD },
                )),
                Cell::from(format!("{}ms", a.total_ms)),
                Cell::from(verdict).style(Style::new().fg(c).add_modifier(Modifier::BOLD)),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Min(8),
        ],
    )
    .header(
        Row::new(vec!["ask", "adm/cand", "top noul", "elapsed", "verdict"])
            .style(Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)),
    )
    .block(block("recent asks"));

    f.render_widget(table, area);
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
                "RETRY" => WARN,
                "WITHHELD" => MAUVE,
                _ => DIM,
            };
            ListItem::new(Line::from(vec![
                Span::styled(format!(" {kind:<16}"), Style::new().fg(c).add_modifier(Modifier::BOLD)),
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
        app.ingest(r#"{"message":"stage ok","stage":"embed","ms":42,"ask_id":"abc"}"#);
        app.ingest(r#"{"message":"stage failed","stage":"admit","ms":900,"ask_id":"abc","error":"boom"}"#);
        app.ingest(
            r#"{"message":"admission complete","ask_id":"abc","candidates":32,"admitted":0,"threshold":0.15,"top_noul":0.04}"#,
        );

        assert_eq!(app.stages["embed"].last(), 42);
        assert_eq!(app.stages["admit"].fails, 1);
        assert_eq!(app.errors, 1);
        assert_eq!(app.total_asks, 1);
        assert_eq!(app.empty_admits, 1, "admitted=0 is an empty admit");
        // 42 + 900 accumulated before the ask completed.
        assert_eq!(app.asks[0].total_ms, 942);
        assert!(app.ingest_noop_on_garbage());
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
            t.draw(|f| super::draw(f, &app)).unwrap();
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
