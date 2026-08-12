//! Interactive `<name>, <quant>` picker over the published LeapBundles.
//!
//! Reached from `cera chat` alone, and only when it is given no `--model`, no
//! `--bundle-id` and no `--quant`, on a terminal. Naming any one of the three
//! is a statement about which model to load, so the picker stays out of the
//! way and the usual error explains what is missing. Without it that
//! no-source-at-all combination is an error telling the user to go read
//! `cera list-bundles` and come back, which is a poor answer when the catalog
//! is one request away.
//!
//! Deliberately not wired into `resolve_engine`, which every model-taking
//! command shares: a picker is right for `chat` and surprising for a one-shot
//! `run --prompt`, and probing the tty down there also changed what
//! `resolve_engine`'s own unit tests do. See `pick_bundle_interactively`.
//!
//! Unlike [`crate::chat_tui`] this takes the alternate screen. The chat TUI
//! deliberately does not, because its whole point is that the conversation
//! lands in scrollback; a picker is transient and leaving a half-scrolled list
//! of 29 bundles behind would be noise, so it restores the screen on exit and
//! the caller prints the one line that matters.
//!
//! Two stages: choose the bundle, then choose its quantization. Typing filters
//! the current stage by substring, which is faster than paging through the list
//! and is the behaviour anyone who has used a fuzzy picker expects.

use std::io::{self, Stdout, Write};
use std::time::Duration;

use anyhow::Result;
use cera::bundle::LeapBundleEntry;

use crate::display_bundle_id;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

/// Restores the terminal on the way out, including on panic.
///
/// Without this a panic mid-render leaves the shell in raw mode inside the
/// alternate screen, which needs a blind `reset` to recover from.
struct ScreenGuard;

impl Drop for ScreenGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen, crossterm::cursor::Show);
    }
}

/// Which of the two lists is on screen.
enum Stage {
    /// Choosing the bundle.
    Bundle,
    /// Choosing a quantization of `bundles[bundle]`.
    Quant { bundle: usize },
}

struct PickerState<'a> {
    bundles: &'a [LeapBundleEntry],
    stage: Stage,
    /// Substring filter for the current stage. Cleared on every stage change,
    /// since a filter typed against bundle names is meaningless against quants.
    filter: String,
    /// Index into the *filtered* list, not into `bundles`.
    cursor: usize,
    /// First visible row, tracked so the cursor can be kept on screen without
    /// re-deriving it from scratch on every key.
    offset: usize,
}

impl<'a> PickerState<'a> {
    fn new(bundles: &'a [LeapBundleEntry]) -> Self {
        Self {
            bundles,
            stage: Stage::Bundle,
            filter: String::new(),
            cursor: 0,
            offset: 0,
        }
    }

    /// The rows currently shown: `(display text, index into the source list)`.
    ///
    /// The single source of truth for what is on screen, used by both the
    /// renderer and [`Self::settle`]. A cheaper count-only twin was tried and
    /// removed: it duplicated the filter for both stages and needed its own
    /// test to stop the two drifting, to save rebuilding a 30-element list at
    /// human-keystroke rate.
    fn rows(&self) -> Vec<(String, usize)> {
        let needle = self.filter.to_ascii_lowercase();
        let matches = |s: &str| needle.is_empty() || s.to_ascii_lowercase().contains(&needle);
        match self.stage {
            Stage::Bundle => self
                .bundles
                .iter()
                .enumerate()
                .filter(|(_, e)| matches(display_bundle_id(&e.name)))
                .map(|(i, e)| {
                    (
                        format!("{}  ({})", display_bundle_id(&e.name), e.quants.join(" ")),
                        i,
                    )
                })
                .collect(),
            Stage::Quant { bundle } => self.bundles[bundle]
                .quants
                .iter()
                .enumerate()
                .filter(|(_, q)| matches(q))
                .map(|(i, q)| (q.clone(), i))
                .collect(),
        }
    }

    /// Clamps the cursor into range and scrolls to keep it visible.
    ///
    /// Called after every change to the state AND after every change to
    /// `height`: filtering can shrink the list out from under a cursor that was
    /// valid a keystroke ago, and a terminal resize moves the window without
    /// touching the state at all.
    fn settle(&mut self, height: usize) {
        let len = self.rows().len();
        if len == 0 {
            self.cursor = 0;
            self.offset = 0;
            return;
        }
        self.cursor = self.cursor.min(len - 1);
        if self.cursor < self.offset {
            self.offset = self.cursor;
        } else if height > 0 && self.cursor >= self.offset + height {
            self.offset = self.cursor + 1 - height;
        }
        // A shrinking list can leave the window past the end.
        self.offset = self.offset.min(len.saturating_sub(height.max(1)));
    }
}

/// Runs the picker. `Ok(None)` means the user cancelled.
///
/// Returns the bundle id **as the catalog spells it** (with its `-GGUF`
/// suffix), so the caller can pass it straight to the loader without a
/// normalization round trip.
pub(crate) fn pick(bundles: &[LeapBundleEntry]) -> Result<Option<(String, String)>> {
    if bundles.is_empty() {
        anyhow::bail!("the LeapBundles catalog came back empty");
    }
    enable_raw_mode()?;
    // Installed before anything that can fail, so every path below restores.
    let _guard = ScreenGuard;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout.by_ref());
    let mut terminal = Terminal::new(backend)?;
    // The cursor is restored by `ScreenGuard`, on the main screen where it
    // actually matters; doing it here as well would only affect the alternate
    // screen this is about to leave.
    run(&mut terminal, bundles)
}

fn run(
    terminal: &mut Terminal<CrosstermBackend<&mut Stdout>>,
    bundles: &[LeapBundleEntry],
) -> Result<Option<(String, String)>> {
    let mut state = PickerState::new(bundles);
    // The viewport height as of the last `settle`. A resize changes the height
    // with no state mutation and no key event, so it has to be reconciled
    // against the drawn geometry rather than on the event path: shrinking the
    // terminal after paging down would otherwise leave the cursor outside the
    // window, showing no selection marker while Enter still picked the row the
    // user could no longer see.
    let mut settled_height = 0usize;
    loop {
        let mut list_height = 0usize;
        terminal.draw(|frame| {
            let area = frame.area();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(1),
                    Constraint::Length(1),
                ])
                .split(area);

            let (title, hint) = match state.stage {
                Stage::Bundle => (
                    "Choose a model".to_string(),
                    "↑↓ move · ⏎ select · type to filter · Esc cancel",
                ),
                Stage::Quant { bundle } => (
                    format!(
                        "Choose a quantization: {}",
                        display_bundle_id(&bundles[bundle].name)
                    ),
                    "↑↓ move · ⏎ select · Esc back",
                ),
            };

            let header = Paragraph::new(Line::from(vec![
                Span::raw("filter: "),
                Span::styled(
                    if state.filter.is_empty() {
                        "(none)".to_string()
                    } else {
                        state.filter.clone()
                    },
                    Style::default().add_modifier(Modifier::BOLD),
                ),
            ]))
            .block(Block::default().borders(Borders::ALL).title(title));
            frame.render_widget(header, chunks[0]);

            list_height = chunks[1].height as usize;
            let rows = state.rows();
            let lines: Vec<Line> = if rows.is_empty() {
                vec![Line::from(Span::raw("  no matches"))]
            } else {
                rows.iter()
                    .skip(state.offset)
                    .take(list_height)
                    .enumerate()
                    .map(|(i, (text, _))| {
                        let selected = state.offset + i == state.cursor;
                        let style = if selected {
                            Style::default().add_modifier(Modifier::REVERSED)
                        } else {
                            Style::default()
                        };
                        Line::from(Span::styled(
                            format!("{} {text}", if selected { ">" } else { " " }),
                            style,
                        ))
                    })
                    .collect()
            };
            frame.render_widget(Paragraph::new(lines), chunks[1]);
            frame.render_widget(Paragraph::new(Line::from(Span::raw(hint))), chunks[2]);
        })?;

        if list_height != settled_height {
            state.settle(list_height);
            settled_height = list_height;
            // Straight back to the top: the frame just drawn used the old
            // offset, so redraw before waiting on a key rather than leaving a
            // stale selection on screen for up to a poll interval.
            continue;
        }

        // Poll rather than block so a resize is picked up promptly; without a
        // timeout the list would keep its old geometry until the next keypress.
        // Nothing is settled on the timeout path itself: only a key or a resize
        // can invalidate the cursor, and the resize case is handled above.
        if !event::poll(Duration::from_millis(200))? {
            continue;
        }
        let Event::Key(KeyEvent {
            code,
            modifiers,
            kind,
            ..
        }) = event::read()?
        else {
            continue;
        };
        // Windows reports press AND release; acting on both double-steps.
        if kind != KeyEventKind::Press {
            continue;
        }
        if modifiers.contains(KeyModifiers::CONTROL) && matches!(code, KeyCode::Char('c')) {
            return Ok(None);
        }
        match code {
            KeyCode::Esc => match state.stage {
                // Esc backs out one stage rather than quitting outright: having
                // to restart the picker because you opened the wrong model is
                // the kind of thing that makes people stop using it.
                Stage::Bundle => return Ok(None),
                Stage::Quant { .. } => {
                    state.stage = Stage::Bundle;
                    state.filter.clear();
                    state.cursor = 0;
                    state.offset = 0;
                }
            },
            KeyCode::Up => state.cursor = state.cursor.saturating_sub(1),
            KeyCode::Down => state.cursor += 1,
            KeyCode::PageUp => state.cursor = state.cursor.saturating_sub(list_height.max(1)),
            KeyCode::PageDown => state.cursor += list_height.max(1),
            KeyCode::Home => state.cursor = 0,
            KeyCode::End => state.cursor = usize::MAX,
            KeyCode::Backspace => {
                state.filter.pop();
            }
            KeyCode::Enter => {
                let rows = state.rows();
                let Some(&(_, index)) = rows.get(state.cursor) else {
                    continue;
                };
                match state.stage {
                    Stage::Bundle => {
                        // Skip the second stage when there is nothing to
                        // choose between.
                        if bundles[index].quants.len() == 1 {
                            return Ok(Some((
                                bundles[index].name.clone(),
                                bundles[index].quants[0].clone(),
                            )));
                        }
                        state.stage = Stage::Quant { bundle: index };
                        state.filter.clear();
                        state.cursor = 0;
                        state.offset = 0;
                    }
                    Stage::Quant { bundle } => {
                        return Ok(Some((
                            bundles[bundle].name.clone(),
                            bundles[bundle].quants[index].clone(),
                        )));
                    }
                }
            }
            // Only an unmodified character types into the filter. Without the
            // guard, Ctrl+U / Ctrl+W / Alt+D and friends insert a literal
            // character instead of being ignored, which reads as the picker
            // losing the list for no reason. ALT rather than checking every
            // modifier: SHIFT is how capitals arrive and must pass through.
            KeyCode::Char(c)
                if !modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                state.filter.push(c);
                // A narrowed list almost never wants to keep its old position.
                state.cursor = 0;
                state.offset = 0;
            }
            _ => {}
        }
        state.settle(list_height);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog() -> Vec<LeapBundleEntry> {
        vec![
            LeapBundleEntry {
                name: "LFM2-1.2B-GGUF".into(),
                quants: vec!["Q4_0".into(), "Q8_0".into()],
            },
            LeapBundleEntry {
                name: "LFM2.5-350M-Instruct-GGUF".into(),
                quants: vec!["Q4_K_M".into()],
            },
            LeapBundleEntry {
                name: "Qwen3-1.7B-GGUF".into(),
                quants: vec!["Q4_0".into()],
            },
        ]
    }

    /// The bundle stage lists every entry, with the `-GGUF` suffix stripped so
    /// what is shown matches what `--bundle-id` takes.
    #[test]
    fn bundle_rows_strip_the_gguf_suffix() {
        let entries = catalog();
        let state = PickerState::new(&entries);
        let rows = state.rows();
        assert_eq!(rows.len(), 3);
        assert!(
            rows[0].0.starts_with("LFM2-1.2B  ("),
            "expected a stripped name with its quants; got {:?}",
            rows[0].0
        );
    }

    /// Filtering is a case-insensitive substring match over the displayed name.
    #[test]
    fn filter_matches_case_insensitively() {
        let entries = catalog();
        let mut state = PickerState::new(&entries);
        state.filter = "qwen".into();
        let rows = state.rows();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1, 2, "the row must carry the index into `bundles`");
    }

    /// A filter matching nothing yields no rows rather than the whole list,
    /// and settling leaves the cursor somewhere renderable.
    #[test]
    fn filter_can_match_nothing() {
        let entries = catalog();
        let mut state = PickerState::new(&entries);
        state.filter = "nonexistent".into();
        assert!(state.rows().is_empty());
        state.settle(10);
        assert_eq!(state.cursor, 0);
        assert_eq!(state.offset, 0);
    }

    /// The quant stage lists the chosen bundle's quants, and its filter applies
    /// to those rather than to the bundle names.
    #[test]
    fn quant_stage_lists_that_bundles_quants() {
        let entries = catalog();
        let mut state = PickerState::new(&entries);
        state.stage = Stage::Quant { bundle: 0 };
        assert_eq!(
            state
                .rows()
                .iter()
                .map(|(t, _)| t.clone())
                .collect::<Vec<_>>(),
            vec!["Q4_0".to_string(), "Q8_0".to_string()]
        );
        state.filter = "q8".into();
        let rows = state.rows();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1, 1);
    }

    /// A cursor left past the end by a shrinking filter is pulled back into
    /// range, and the window follows it. Without this the picker renders an
    /// empty viewport and Enter selects nothing.
    #[test]
    fn settle_clamps_a_stale_cursor() {
        let entries = catalog();
        let mut state = PickerState::new(&entries);
        state.cursor = 99;
        state.settle(2);
        assert_eq!(state.cursor, 2, "clamped to the last row");
        assert!(
            state.offset <= state.cursor,
            "the window must contain the cursor; offset={} cursor={}",
            state.offset,
            state.cursor
        );
    }

    /// Scrolling down past the viewport advances the window by exactly enough
    /// to keep the cursor on the last visible row.
    #[test]
    fn settle_scrolls_to_follow_the_cursor() {
        let entries = catalog();
        let mut state = PickerState::new(&entries);
        state.cursor = 2;
        state.settle(2);
        assert_eq!(state.offset, 1, "cursor 2 with height 2 shows rows 1..=2");
    }

    /// A viewport that SHRINKS pulls the window down to keep the cursor
    /// visible, with no state change of its own.
    ///
    /// The regression this guards: `settle` was called only on key events, so
    /// a terminal resized after paging down left the cursor outside the drawn
    /// window. The list then showed no selection marker at all while Enter
    /// still selected the row the user could no longer see. The run loop now
    /// reconciles against the drawn height instead.
    #[test]
    fn settle_follows_a_shrinking_viewport() {
        let entries = catalog();
        let mut state = PickerState::new(&entries);
        // Bottom row, roomy terminal: everything fits, so no scrolling yet.
        state.cursor = 2;
        state.settle(3);
        assert_eq!(state.offset, 0);
        // Now the terminal shrinks to one row with no key pressed.
        state.settle(1);
        assert_eq!(
            state.offset, 2,
            "the window must move to contain the cursor after a resize"
        );
    }
}
