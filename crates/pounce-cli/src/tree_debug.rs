//! Interactive **tree** debugger for the spatial branch-and-bound global
//! solver (`pounce --solver global --debug`).
//!
//! Branch-and-bound is a tree search, so this is a different REPL from the
//! interior-point [`debug_repl`](crate::debug_repl): it pauses at tree
//! checkpoints (node selection, relaxation, incumbent, prune, branch) and
//! exposes the node box, the global lower/upper bounds, the optimality gap,
//! and the frontier — rather than an iterate. It drives the solver through
//! the shared [`TreeDebugHook`] surface.
//!
//! Commands (text REPL; `--debug-script` feeds the same commands from a
//! file, auto-continuing once the script is exhausted):
//!
//! ```text
//!   s | step          run to the next checkpoint
//!   c | continue      run until a breakpoint (or the end)
//!   node              the current node's box and bound
//!   bounds            global lower bound, incumbent, gap
//!   gap               the optimality gap
//!   incumbent | inc   the best feasible point so far
//!   frontier          number of open nodes
//!   info | i          re-print the pause banner
//!   break incumbent             stop when the incumbent improves
//!   break gap <x>               stop when the gap ≤ x
//!   break depth <n>             stop at depth ≥ n
//!   break node <id>             stop when node #id is selected
//!   help | h          this list
//!   q | quit          stop the search now
//! ```

use crate::cli::DebugMode;
use crate::debug_repl::SharedScript;
use pounce_common::debug::{DebugAction, TreeCheckpoint, TreeDebugHook, TreeDebugState};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::path::Path;
use std::rc::Rc;

/// A breakpoint condition for the tree search.
enum TreeBreak {
    /// Pause when a new incumbent (better feasible point) is found.
    Incumbent,
    /// Pause once the optimality gap drops to or below this value.
    Gap(f64),
    /// Pause when a node at this depth (or deeper) is selected.
    Depth(usize),
    /// Pause when the node with this id is selected.
    Node(u64),
}

impl TreeBreak {
    fn matches(&self, st: &dyn TreeDebugState) -> bool {
        match self {
            TreeBreak::Incumbent => st.checkpoint() == TreeCheckpoint::IncumbentFound,
            TreeBreak::Gap(x) => st.gap() <= *x,
            TreeBreak::Depth(n) => {
                st.checkpoint() == TreeCheckpoint::NodeSelected && st.depth() >= *n
            }
            TreeBreak::Node(id) => {
                st.checkpoint() == TreeCheckpoint::NodeSelected && st.node_id() == *id
            }
        }
    }
}

/// Outcome of dispatching one REPL command.
enum Flow {
    /// Keep reading commands at this pause.
    Stay,
    /// Resume the search.
    Resume,
    /// Stop the search.
    Stop,
}

/// The branch-and-bound REPL. Implements [`TreeDebugHook`].
pub struct TreeDebugger {
    mode: DebugMode,
    /// Scripted commands, shared with the interior-point sub-solve debugger
    /// so one `--debug-script` interleaves tree and `into` commands.
    script: Option<SharedScript>,
    editor: Option<rustyline::DefaultEditor>,
    /// Pause at the next checkpoint (one-shot; set by `step` and at start).
    step: bool,
    breaks: Vec<TreeBreak>,
    quit: bool,
    /// JSON mode only: whether the one-time `hello` handshake has been
    /// emitted yet (sent lazily before the first `tree_pause`).
    hello_sent: bool,
}

impl TreeDebugger {
    pub fn new(mode: DebugMode) -> Self {
        Self {
            mode,
            script: None,
            editor: None,
            // Enter paused at the first checkpoint, like a debugger breaking in.
            step: true,
            breaks: Vec::new(),
            quit: false,
            hello_sent: false,
        }
    }

    /// Emit the one-time JSON handshake, mirroring the iteration-loop
    /// debugger's `hello` (see `debug_repl::SolverDebugger::emit_hello`) but
    /// for the spatial branch-and-bound *tree*. A B&B node has no iterate, so
    /// this is a deliberately **distinct** protocol variant — `kind: "tree"`,
    /// pauses arrive as `tree_pause` (not `pause`), and there is no
    /// request/result command loop (`request_ids: false`). Advertising this up
    /// front lets a generic `pounce-dbg/1` client recognise the tree backend
    /// instead of blocking forever waiting for an iteration-loop `hello`+`pause`.
    fn emit_hello(&self) {
        println!(
            "{}",
            serde_json::to_string(&Self::hello_event()).expect("hello serializes")
        );
    }

    /// The `hello` payload (split out so a test can pin the wire contract).
    fn hello_event() -> serde_json::Value {
        serde_json::json!({
            "event": "hello",
            "protocol": "pounce-dbg/1",
            "kind": "tree",
            "pounce_version": env!("CARGO_PKG_VERSION"),
            "pause_event": "tree_pause",
            "capabilities": {
                "request_ids": false,
                "step_into_relaxation": true,
            },
            "checkpoints": [
                "node_selected", "relaxation_solved", "incumbent_found",
                "node_pruned", "branched", "terminated",
            ],
            "commands": [
                "help", "info", "step", "continue", "into", "node", "bounds",
                "gap", "incumbent", "frontier", "break", "quit",
            ],
            "metrics": [
                "node", "depth", "node_lb", "global_lb", "incumbent", "gap",
                "frontier", "nodes",
            ],
        })
    }

    /// Feed commands from a script file (one per line; `#` comments and blank
    /// lines ignored). Once exhausted the search auto-continues. The queue is
    /// shared (see [`shared_script`](Self::shared_script)) so a stepped-into
    /// relaxation reads its interior-point commands from the same script.
    pub fn with_script(mut self, path: &Path) -> Self {
        let mut q = VecDeque::new();
        match std::fs::read_to_string(path) {
            Ok(text) => {
                for line in text.lines() {
                    let l = line.trim();
                    if !l.is_empty() && !l.starts_with('#') {
                        q.push_back(l.to_string());
                    }
                }
            }
            Err(e) => eprintln!("pounce: cannot read debug script {}: {e}", path.display()),
        }
        self.script = Some(Rc::new(RefCell::new(q)));
        self
    }

    /// The shared command queue (when scripted), to hand to the sub-solve
    /// interior-point debugger so `into` replays scripted IP commands.
    pub fn shared_script(&self) -> Option<SharedScript> {
        self.script.clone()
    }

    fn read_command(&mut self) -> Option<String> {
        if let Some(q) = &self.script {
            let c = q.borrow_mut().pop_front();
            if let Some(cmd) = &c {
                println!("[script] {cmd}");
            }
            c
        } else {
            if self.editor.is_none() {
                self.editor = rustyline::DefaultEditor::new().ok();
            }
            match self.editor.as_mut() {
                Some(ed) => match ed.readline("(btree) ") {
                    Ok(line) => {
                        let _ = ed.add_history_entry(line.as_str());
                        Some(line)
                    }
                    // Ctrl-C / Ctrl-D / read error: resume rather than hang.
                    Err(_) => None,
                },
                None => None,
            }
        }
    }

    fn dispatch(&mut self, line: &str, st: &mut dyn TreeDebugState) -> Flow {
        // Accept both the plain text form (`quit`) and the JSON command
        // framing the iteration-loop protocol uses (`{"cmd":"quit","id":N}`).
        // The tree debugger doesn't run a request/result loop, but honouring
        // the JSON form means a generic `pounce-dbg/1` client's `quit` /
        // `continue` still drives it — notably the MCP proxy's `close()`,
        // which sends `{"cmd":"quit"}` and would otherwise wait out its
        // process-kill timeout.
        let normalized = normalize_command(line);
        let mut toks = normalized.split_whitespace();
        let Some(cmd) = toks.next() else {
            return Flow::Stay;
        };
        match cmd {
            "s" | "step" => {
                self.step = true;
                Flow::Resume
            }
            "c" | "continue" => Flow::Resume,
            "into" => {
                // Step into this node's relaxation solve with the
                // interior-point debugger (only meaningful at NodeSelected).
                if st.checkpoint() == TreeCheckpoint::NodeSelected {
                    st.request_subsolve_debug();
                    println!("stepping into node #{}'s relaxation solve…", st.node_id());
                    Flow::Resume
                } else {
                    println!("`into` works at a node_selected pause (before the relaxation)");
                    Flow::Stay
                }
            }
            "q" | "quit" => Flow::Stop,
            "node" => {
                let (lo, hi) = st.node_box();
                println!(
                    "node #{}  depth {}  lb={:.8e}",
                    st.node_id(),
                    st.depth(),
                    st.node_lb()
                );
                for (i, (l, h)) in lo.iter().zip(&hi).enumerate() {
                    println!("  x[{i}] ∈ [{l:.6e}, {h:.6e}]   (width {:.3e})", h - l);
                }
                Flow::Stay
            }
            "bounds" => {
                let inc = st
                    .incumbent()
                    .map(|v| format!("{v:.8e}"))
                    .unwrap_or_else(|| "none".into());
                println!(
                    "lower={:.8e}  incumbent(upper)={inc}  gap={:.3e}",
                    st.global_lb(),
                    st.gap()
                );
                Flow::Stay
            }
            "gap" => {
                println!("gap = {:.6e}", st.gap());
                Flow::Stay
            }
            "incumbent" | "inc" => {
                match (st.incumbent(), st.incumbent_point()) {
                    (Some(obj), Some(x)) => {
                        println!("incumbent obj = {obj:.8e}  at x = {}", fmt_vec(&x));
                    }
                    _ => println!("no incumbent yet"),
                }
                Flow::Stay
            }
            "frontier" => {
                println!("frontier: {} open node(s)", st.frontier_len());
                Flow::Stay
            }
            "info" | "i" => {
                self.print_status(st);
                Flow::Stay
            }
            "break" | "b" => {
                self.cmd_break(toks.next(), toks.next());
                Flow::Stay
            }
            "help" | "h" => {
                print_help();
                Flow::Stay
            }
            other => {
                println!("unknown command `{other}` (try `help`)");
                Flow::Stay
            }
        }
    }

    fn cmd_break(&mut self, what: Option<&str>, arg: Option<&str>) {
        match what {
            Some("incumbent") => {
                self.breaks.push(TreeBreak::Incumbent);
                println!("breakpoint: incumbent improvement");
            }
            Some("gap") => match arg.and_then(|a| a.parse::<f64>().ok()) {
                Some(x) => {
                    self.breaks.push(TreeBreak::Gap(x));
                    println!("breakpoint: gap ≤ {x:.3e}");
                }
                None => println!("usage: break gap <value>"),
            },
            Some("depth") => match arg.and_then(|a| a.parse::<usize>().ok()) {
                Some(n) => {
                    self.breaks.push(TreeBreak::Depth(n));
                    println!("breakpoint: depth ≥ {n}");
                }
                None => println!("usage: break depth <n>"),
            },
            Some("node") => match arg.and_then(|a| a.parse::<u64>().ok()) {
                Some(id) => {
                    self.breaks.push(TreeBreak::Node(id));
                    println!("breakpoint: node #{id}");
                }
                None => println!("usage: break node <id>"),
            },
            _ => println!("usage: break incumbent | gap <x> | depth <n> | node <id>"),
        }
    }

    fn print_status(&self, st: &dyn TreeDebugState) {
        if matches!(self.mode, DebugMode::Json) {
            // Compact machine-readable status line.
            let inc = st
                .incumbent()
                .map(|v| v.to_string())
                .unwrap_or_else(|| "null".into());
            println!(
                "{{\"event\":\"tree_pause\",\"checkpoint\":\"{}\",\"node\":{},\"depth\":{},\
                 \"node_lb\":{},\"global_lb\":{},\"incumbent\":{inc},\"gap\":{},\
                 \"frontier\":{},\"nodes\":{}}}",
                st.checkpoint().as_str(),
                st.node_id(),
                st.depth(),
                fnum(st.node_lb()),
                fnum(st.global_lb()),
                fnum(st.gap()),
                st.frontier_len(),
                st.nodes_processed(),
            );
            return;
        }
        if st.checkpoint() == TreeCheckpoint::Terminated {
            println!(
                "── btree ── TERMINATED ({})  nodes={}  lower={:.8e}  incumbent={}  gap={:.3e}",
                st.status().unwrap_or("?"),
                st.nodes_processed(),
                st.global_lb(),
                st.incumbent()
                    .map(|v| format!("{v:.8e}"))
                    .unwrap_or_else(|| "none".into()),
                st.gap(),
            );
            return;
        }
        let inc = st
            .incumbent()
            .map(|v| format!("{v:.6e}"))
            .unwrap_or_else(|| "none".into());
        println!(
            "── btree ── {} node #{} depth {}  lb={:.6e}  inc={inc}  gap={:.3e}  frontier={} (nodes {})",
            st.checkpoint().as_str(),
            st.node_id(),
            st.depth(),
            st.node_lb(),
            st.gap(),
            st.frontier_len(),
            st.nodes_processed(),
        );
        if let Some(r) = st.prune_reason() {
            println!("   pruned: {}", r.as_str());
        }
        if let Some(v) = st.branch_var() {
            println!("   branching on x[{v}]");
        }
    }
}

impl TreeDebugHook for TreeDebugger {
    fn at_node(&mut self, st: &mut dyn TreeDebugState) -> DebugAction {
        if self.quit {
            return DebugAction::Stop;
        }
        let terminal = st.checkpoint() == TreeCheckpoint::Terminated;
        let mut pause = self.step || terminal;
        if !pause {
            pause = self.breaks.iter().any(|b| b.matches(st));
        }
        if !pause {
            return DebugAction::Resume;
        }
        self.step = false; // consume the one-shot step
        if matches!(self.mode, DebugMode::Json) && !self.hello_sent {
            self.emit_hello();
            self.hello_sent = true;
        }
        self.print_status(st);

        loop {
            let Some(cmd) = self.read_command() else {
                // Script exhausted / interactive EOF: resume the search.
                return DebugAction::Resume;
            };
            match self.dispatch(&cmd, st) {
                Flow::Stay => continue,
                Flow::Resume => return DebugAction::Resume,
                Flow::Stop => {
                    self.quit = true;
                    return DebugAction::Stop;
                }
            }
        }
    }
}

fn print_help() {
    println!(
        "tree-debugger commands:\n\
         \x20 s|step            run to the next checkpoint\n\
         \x20 c|continue        run until a breakpoint or the end\n\
         \x20 into              step into this node's relaxation (IP debugger)\n\
         \x20 node              current node box and bound\n\
         \x20 bounds            global lower bound, incumbent, gap\n\
         \x20 gap               optimality gap\n\
         \x20 incumbent|inc     best feasible point so far\n\
         \x20 frontier          number of open nodes\n\
         \x20 info|i            re-print the pause banner\n\
         \x20 break incumbent | gap <x> | depth <n> | node <id>\n\
         \x20 q|quit            stop the search now"
    );
}

fn fmt_vec(v: &[f64]) -> String {
    const MAX: usize = 12;
    if v.len() <= MAX {
        let parts: Vec<String> = v.iter().map(|x| format!("{x:.6e}")).collect();
        format!("[{}]", parts.join(", "))
    } else {
        let head: Vec<String> = v[..MAX].iter().map(|x| format!("{x:.6e}")).collect();
        format!("[{}, … ({} total)]", head.join(", "), v.len())
    }
}

/// Map a raw input line to a debugger command. A line that is a JSON object
/// with a string `cmd` field (the iteration-loop protocol's command framing,
/// `{"cmd":"continue","id":3}`) yields that `cmd`; anything else passes
/// through unchanged so the plain text REPL form still works.
fn normalize_command(line: &str) -> String {
    let t = line.trim();
    if t.starts_with('{') {
        if let Ok(serde_json::Value::Object(m)) = serde_json::from_str::<serde_json::Value>(t) {
            if let Some(c) = m.get("cmd").and_then(|v| v.as_str()) {
                return c.to_string();
            }
        }
    }
    line.to_string()
}

/// Format a finite f64 for JSON, mapping non-finite to `null`.
fn fnum(x: f64) -> String {
    if x.is_finite() {
        x.to_string()
    } else {
        "null".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tree debugger's JSON `hello` is a *distinct* protocol variant from
    /// the iteration-loop debugger's: it must announce `kind: "tree"`, name its
    /// `tree_pause` pause event, and disclaim the request/result loop
    /// (`request_ids: false`). The MCP `debug_start` proxy keys on exactly
    /// these to reject the global backend fast instead of hanging — keep this
    /// contract in lockstep with `studio/mcp/.../server.py::read_startup`.
    #[test]
    fn tree_hello_announces_a_distinct_protocol_variant() {
        let h = TreeDebugger::hello_event();
        assert_eq!(h["event"], "hello");
        assert_eq!(h["protocol"], "pounce-dbg/1");
        assert_eq!(h["kind"], "tree");
        assert_eq!(h["pause_event"], "tree_pause");
        assert_eq!(h["capabilities"]["request_ids"], false);
        // Metrics must match the fields every `tree_pause` actually carries.
        let metrics: Vec<&str> = h["metrics"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        for m in [
            "node",
            "depth",
            "node_lb",
            "global_lb",
            "incumbent",
            "gap",
            "frontier",
            "nodes",
        ] {
            assert!(metrics.contains(&m), "hello.metrics missing {m}");
        }
    }

    #[test]
    fn normalize_command_accepts_json_and_plain_forms() {
        // JSON command framing (what the MCP proxy's close() sends).
        assert_eq!(normalize_command(r#"{"cmd":"quit","id":7}"#), "quit");
        assert_eq!(normalize_command(r#"  {"cmd":"continue"}  "#), "continue");
        // Plain text passes through unchanged.
        assert_eq!(normalize_command("step"), "step");
        assert_eq!(normalize_command("break depth 3"), "break depth 3");
        // Malformed / non-object JSON falls back to the raw line.
        assert_eq!(normalize_command("{not json"), "{not json");
    }
}
