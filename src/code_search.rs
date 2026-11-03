//! Read-only access to one source tree for the LLM (`/llm/codeSearch`): the tools
//! `list_files`, `search_text` and `read_file`, offered to the model as OpenAI function tools.
//!
//! Every path is resolved inside the canonical root (`..` and symlinks leading out are refused);
//! hidden entries, denied dirs/files/extensions, binaries and big files are invisible.
//! Tool output is capped, since it goes into the model's context.

use std::{
    fs,
    path::{Component, Path, PathBuf},
};

use serde_derive::Deserialize;
use serde_json::{Value, json};

/// Files bigger than this are skipped (generated or data files, not sources).
const MAX_FILE_BYTES: u64 = 512 * 1024;
/// Caps of one tool result.
const MAX_LIST_ENTRIES: usize = 300;
const MAX_SEARCH_MATCHES: usize = 30;
const MAX_FILES_VISITED: usize = 20_000;
const MAX_LINE_CHARS: usize = 200;
const MAX_READ_LINES: usize = 120;
// ~1.5k tokens: several results must fit the model context (8k on a small local model)
const MAX_OUTPUT_CHARS: usize = 6_000;
/// Cap of a file shown to the user (`show_file`), it goes into the stored answer, not the model context.
const MAX_SHOWN_CHARS: usize = 100_000;

/// Result of one tool call.
pub struct ToolOutput {
    /// Tool message for the model.
    pub for_model: String,
    /// Markdown attached to the final answer as is (`show_file`).
    pub shown: Option<String>,
}

/// `/llm/codeSearch` config section.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodeSearchConfig {
    /// Root of the tree the model may read.
    pub root: PathBuf,
    /// Tool rounds per answer; after them the model must answer without tools.
    pub max_steps: u32,
    /// Directory names hidden anywhere in the tree (e.g. `target`).
    #[serde(default)]
    pub deny_dirs: Vec<String>,
    /// File names hidden anywhere in the tree (e.g. `local.json`, which holds credentials).
    #[serde(default)]
    pub deny_files: Vec<String>,
    /// File extensions hidden anywhere in the tree (e.g. `pem`).
    #[serde(default)]
    pub deny_extensions: Vec<String>,
    /// System prompt prepended to every dialog; `{root}` is replaced with the root path.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Archify diagrams the `diagrams` tool shows as a gallery; unset = no such tool.
    #[serde(default)]
    pub diagrams: Option<DiagramsConfig>,
}

/// `/llm/codeSearch/diagrams` config section.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiagramsConfig {
    /// Directory of the diagram sources (`<name>.json` with `meta.title`/`meta.output`), relative to the root.
    pub dir: PathBuf,
    /// Prefix of the diagram urls the page opens, relative to the page (e.g. `diagrams/`).
    pub url_prefix: String,
}

/// Language tag of the fenced block the `diagrams` tool attaches; the page renders it as a gallery.
pub const DIAGRAMS_FENCE_LANG: &str = "skw-diagrams";

const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant with read-only access to the source code of a software project named 'SKW Framework' \
(root: {root}). You CAN read every source file through the tools, so never say you have no access to the code \
 and never guess: for any question about the project or its source code, first call the tools, then answer \
(questions about a person are not about the code, the tools know nothing about people):\n\
- search_text finds a SHORT keyword or identifier (one or two words, e.g. `heartbeat` or `BACK_MESSAGE_TIMEOUT`), never a whole sentence. \
If nothing matches, try other keywords, synonyms or a part of the word.\n\
- list_files shows what is inside a directory.\n\
- read_file reads the lines around a match, so you see the actual code.\n\
- show_file shows a whole file to the user.\n\
{diagrams_tool}\
Make several tool calls when needed. Paths are relative to the project root. Answer from what you read, cite file paths, \
and say so if you could not find something. \
When the user asks to show, print or display a file, call show_file: the file is attached to your answer exactly as it is, \
so never copy or rewrite file contents yourself.\n\n\
About the project: the SKW Framework (also written SKW, skw or skw2; repository skw2) is a Rust microservice framework \
written by Sergey Kutsenko as his own project, and it runs this chat. It is not related to the companies \
or jobs in his resume; never claim it was used there. It is a Cargo workspace:\n\
- libs/shared: the core every service builds on: config loading, Postgres pools, the Iggy message bus client, \
RabbitMQ and Redis clients, and the generic loops services run (Iggy request/reply consumers, RabbitMQ task consumers).\n\
- libs/http-protos and libs/task-protos: the message contracts (Iggy request/reply types, RabbitMQ task payloads).\n\
- gates/api: HTTPS gateway (ntex). gates/ws: WebSocket gateway. Both verify ed25519 request signatures, \
route each request to an Iggy topic and return the service's reply, which comes back on a shared back topic.\n\
- services/auth: API keys (CreateKey, GetKeyById). services/ping: the reference service. services/dialog: these LLM chats \
(dialog-stream-service handles requests, dialog-mq-service asks the LLM, with you as the model). \
services/test: integration-test runner.\n\
- frontend: the resume web page with this chat. frontend/diagrams: architecture diagrams.\n\
Request flow: client -> gate (signature check) -> Iggy topic -> service -> reply on the back topic -> gate -> client; \
slow work goes to RabbitMQ workers. CLAUDE.md in the root describes the architecture in detail.\n\n\
Project layout (depth 2):\n{layout}";

pub struct CodeSearch {
    root: PathBuf,
    max_steps: u32,
    deny_dirs: Vec<String>,
    deny_files: Vec<String>,
    deny_extensions: Vec<String>,
    system_prompt: String,
    /// Canonical diagrams dir and url prefix (`/llm/codeSearch/diagrams`).
    diagrams: Option<(PathBuf, String)>,
}

impl CodeSearch {
    /// Panics if the root doesn't exist (startup only).
    pub fn new(cfg: &CodeSearchConfig) -> Self {
        let root = cfg
            .root
            .canonicalize()
            .unwrap_or_else(|e| panic!("/llm/codeSearch/root {:?}: {e}", cfg.root));
        assert!(
            root.is_dir(),
            "/llm/codeSearch/root {root:?} is not a directory"
        );

        let mut search = CodeSearch {
            root,
            max_steps: cfg.max_steps,
            deny_dirs: cfg.deny_dirs.clone(),
            deny_files: cfg.deny_files.clone(),
            deny_extensions: cfg
                .deny_extensions
                .iter()
                .map(|e| e.trim_start_matches('.').to_lowercase())
                .collect(),
            system_prompt: String::new(),
            diagrams: None,
        };
        search.diagrams = cfg.diagrams.as_ref().map(|d| {
            let dir = search
                .root
                .join(&d.dir)
                .canonicalize()
                .unwrap_or_else(|e| panic!("/llm/codeSearch/diagrams/dir {:?}: {e}", d.dir));
            (dir, d.url_prefix.clone())
        });

        let template = cfg
            .system_prompt
            .as_deref()
            .unwrap_or(DEFAULT_SYSTEM_PROMPT);
        let layout = if template.contains("{layout}") {
            search.call("list_files", r#"{"depth": 2}"#).for_model
        } else {
            String::new()
        };
        search.system_prompt = template
            .replace("{root}", &search.root.display().to_string())
            .replace("{layout}", &layout)
            .replace(
                "{diagrams_tool}",
                if search.diagrams.is_some() {
                    "- diagrams shows the SKW Framework's architecture diagrams (pictures) to the user; use it when the user \
asks for diagrams, schemes, pictures or a visual overview of the architecture.\n"
                } else {
                    ""
                },
            );
        search
    }

    pub fn max_steps(&self) -> u32 {
        self.max_steps
    }

    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// The `tools` array of a chat completions request.
    pub fn tool_definitions(&self) -> Value {
        let mut tools = json!([
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files and directories under a directory of the project (recursively, up to a depth).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dir": { "type": "string", "description": "Directory relative to the project root; empty or '.' for the root." },
                            "depth": { "type": "integer", "description": "How many levels deep to list, 1-4 (default 2)." }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_text",
                    "description": "Case-insensitive plain-text search in the project's text files for a short keyword or identifier (not a sentence). Returns matching lines as path:line: text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string", "description": "One short keyword or identifier, e.g. 'heartbeat' (plain text, not a regex)." },
                            "dir": { "type": "string", "description": "Optional directory to search in, relative to the project root." },
                            "extension": { "type": "string", "description": "Optional file extension filter, e.g. 'rs' or 'md'." }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "show_file",
                    "description": "Show a whole text file of the project to the user, exactly as it is (it is attached to your answer). Use it when the user asks to show, print or display a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": { "type": "string", "description": "File path relative to the project root." }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read lines of a text file of the project, with line numbers (at most 120 lines per call).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": { "type": "string", "description": "File path relative to the project root." },
                            "start_line": { "type": "integer", "description": "First line, 1-based (default 1)." },
                            "end_line": { "type": "integer", "description": "Last line, inclusive (default start_line + 119)." }
                        },
                        "required": ["path"]
                    }
                }
            }
        ]);
        if self.diagrams.is_some()
            && let Some(list) = tools.as_array_mut()
        {
            list.push(json!({
                "type": "function",
                "function": {
                    "name": "diagrams",
                    "description": "Show all architecture diagrams of the SKW Framework to the user as a gallery of previews (a click opens the full diagram). Use it when the user asks for diagrams, schemes or pictures of the architecture.",
                    "parameters": { "type": "object", "properties": {}, "required": [] }
                }
            }));
        }
        tools
    }

    /// A user message `/<tool> <args>` runs that tool directly, without the model, and returns the
    /// whole answer (Markdown); `None` if the message is not a slash command. Blocking (file system).
    ///
    /// `/list_files [dir] [depth]`, `/search_text <text> [dir=<dir>] [ext=<ext>]`,
    /// `/read_file <path> [start] [end]`, `/show_file <path>`, `/diagrams`; `/help` and any other `/word` get the list.
    pub fn run_command(&self, text: &str) -> Option<String> {
        let text = text.trim();
        let rest = text.strip_prefix('/')?;
        let (name, args) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
            return None; // e.g. a path like `/etc/hosts`: an ordinary question
        }
        let words: Vec<&str> = args.split_whitespace().collect();

        let arguments = match name {
            "list_files" => json!({ "dir": words.first().copied().unwrap_or(""), "depth": words.get(1).and_then(|d| d.parse::<u64>().ok()) }),
            "search_text" => {
                let (options, query): (Vec<&str>, Vec<&str>) = words
                    .iter()
                    .partition(|w| w.starts_with("dir=") || w.starts_with("ext="));
                let option = |key: &str| {
                    options
                        .iter()
                        .find_map(|o| o.strip_prefix(key))
                        .unwrap_or("")
                };
                json!({ "query": query.join(" "), "dir": option("dir="), "extension": option("ext=") })
            }
            "read_file" => json!({
                "path": words.first().copied().unwrap_or(""),
                "start_line": words.get(1).and_then(|n| n.parse::<u64>().ok()),
                "end_line": words.get(2).and_then(|n| n.parse::<u64>().ok()),
            }),
            "show_file" => json!({ "path": words.first().copied().unwrap_or("") }),
            "diagrams" if self.diagrams.is_some() => json!({}),
            _ => return Some(self.command_help(name)),
        };

        let output = self.call(name, &arguments.to_string());
        if output.for_model.starts_with("error: ") {
            return Some(format!(
                "`/{name}` failed: {}",
                &output.for_model["error: ".len()..]
            ));
        }
        Some(match name {
            "show_file" | "diagrams" => output.shown.unwrap_or(output.for_model),
            "read_file" => {
                // first line: "<path> (lines a-b of n)", then the numbered lines
                let (header, body) = output
                    .for_model
                    .split_once('\n')
                    .unwrap_or((&output.for_model, ""));
                let lang = Path::new(words.first().copied().unwrap_or(""))
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");
                format!("**{header}**\n\n{}", fenced(body.trim_end(), lang))
            }
            _ => fenced(&output.for_model, ""),
        })
    }

    fn command_help(&self, name: &str) -> String {
        let intro = if name == "help" {
            String::new()
        } else {
            format!("Unknown command `/{name}`. ")
        };
        let mut help = format!(
            "{intro}Commands run a tool directly, without the model:\n\n\
- `/list_files [dir] [depth]`: list a directory (depth 1-4, default 2)\n\
- `/search_text <text> [dir=<dir>] [ext=<ext>]`: find a text in the source files\n\
- `/read_file <path> [start] [end]`: read lines of a file (at most {MAX_READ_LINES})\n\
- `/show_file <path>`: show a whole file"
        );
        if self.diagrams.is_some() {
            help.push_str("\n- `/diagrams`: show the architecture diagrams");
        }
        help
    }

    /// Runs one tool call; failures are returned as text for the model, never as an error.
    /// Blocking (file system), call it from `spawn_blocking`.
    pub fn call(&self, name: &str, arguments: &str) -> ToolOutput {
        let args: Value = match serde_json::from_str(if arguments.trim().is_empty() {
            "{}"
        } else {
            arguments
        }) {
            Ok(args) => args,
            Err(e) => {
                return ToolOutput {
                    for_model: format!("error: arguments are not valid JSON: {e}"),
                    shown: None,
                };
            }
        };
        let str_arg = |key: &str| args.get(key).and_then(Value::as_str).unwrap_or("").trim();
        let int_arg = |key: &str| {
            args.get(key).and_then(|v| {
                v.as_u64()
                    .or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
            })
        };

        let result = match name {
            "list_files" => self.list_files(
                str_arg("dir"),
                int_arg("depth").unwrap_or(2).clamp(1, 4) as usize,
            ),
            "search_text" => self.search_text(str_arg("query"), str_arg("dir"), str_arg("extension")),
            "read_file" => self.read_file(str_arg("path"), int_arg("start_line"), int_arg("end_line")),
            "diagrams" if self.diagrams.is_some() => {
                return match self.diagram_gallery() {
                    Ok((for_model, shown)) => ToolOutput {
                        for_model,
                        shown: Some(shown),
                    },
                    Err(e) => ToolOutput {
                        for_model: format!("error: {e}"),
                        shown: None,
                    },
                };
            }
            "show_file" => {
                return match self.show_file(str_arg("path")) {
                    Ok((for_model, shown)) => ToolOutput {
                        for_model,
                        shown: Some(shown),
                    },
                    Err(e) => ToolOutput {
                        for_model: format!("error: {e}"),
                        shown: None,
                    },
                };
            }
            other => Err(format!("unknown tool '{other}'")),
        };

        let mut out = result.unwrap_or_else(|e| format!("error: {e}"));
        if out.len() > MAX_OUTPUT_CHARS {
            let cut = floor_char_boundary(&out, MAX_OUTPUT_CHARS);
            out.truncate(cut);
            out.push_str("\n... (output truncated)");
        }
        ToolOutput {
            for_model: out,
            shown: None,
        }
    }

    /// Canonical path of `rel` inside the root, if it exists and is visible.
    fn resolve(&self, rel: &str) -> Result<PathBuf, String> {
        let rel = rel.trim().trim_start_matches("./");
        let rel_path = Path::new(rel);
        // absolute paths are accepted only when they point inside the root
        let joined = if rel_path.is_absolute() {
            rel_path.to_path_buf()
        } else {
            self.root.join(rel_path)
        };

        let path = joined
            .canonicalize()
            .map_err(|_| format!("'{rel}' not found"))?;
        let inside = path
            .strip_prefix(&self.root)
            .map_err(|_| format!("'{rel}' is outside the project"))?;

        if !self.is_visible(inside, path.is_dir()) {
            return Err(format!("'{rel}' is not accessible"));
        }
        Ok(path)
    }

    /// Checks every component of a root-relative path against the hidden/deny rules.
    fn is_visible(&self, rel: &Path, is_dir: bool) -> bool {
        let names: Vec<&str> = rel
            .components()
            .filter_map(|c| match c {
                Component::Normal(name) => name.to_str(),
                _ => None,
            })
            .collect();

        names.iter().enumerate().all(|(i, name)| {
            let is_last = i + 1 == names.len();
            if name.starts_with('.') {
                return false;
            }
            if !is_last || is_dir {
                !self.deny_dirs.iter().any(|d| d == name)
            } else {
                self.is_visible_file_name(name)
            }
        })
    }

    fn is_visible_file_name(&self, name: &str) -> bool {
        if name.starts_with('.') || self.deny_files.iter().any(|f| f == name) {
            return false;
        }
        match Path::new(name).extension().and_then(|e| e.to_str()) {
            Some(ext) => !self.deny_extensions.contains(&ext.to_lowercase()),
            None => true,
        }
    }

    fn display(&self, path: &Path) -> String {
        path.strip_prefix(&self.root)
            .unwrap_or(path)
            .display()
            .to_string()
    }

    /// Visible children of a directory, sorted; symlinks are skipped.
    fn children(&self, dir: &Path) -> Vec<(PathBuf, bool)> {
        let Ok(entries) = fs::read_dir(dir) else {
            return Vec::new();
        };
        let mut children: Vec<(PathBuf, bool)> = entries
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let file_type = entry.file_type().ok()?;
                if file_type.is_symlink() {
                    return None;
                }
                let name = entry.file_name();
                let name = name.to_str()?;
                let visible = if file_type.is_dir() {
                    !name.starts_with('.') && !self.deny_dirs.iter().any(|d| d == name)
                } else {
                    file_type.is_file() && self.is_visible_file_name(name)
                };
                visible.then(|| (entry.path(), file_type.is_dir()))
            })
            .collect();
        children.sort();
        children
    }

    fn list_files(&self, dir: &str, depth: usize) -> Result<String, String> {
        let start = self.resolve(if dir.is_empty() { "." } else { dir })?;
        if !start.is_dir() {
            return Err(format!("'{dir}' is not a directory"));
        }

        let mut out = Vec::new();
        self.list_into(&start, depth, &mut out);
        if out.is_empty() {
            return Ok("(empty)".to_string());
        }
        Ok(out.join("\n"))
    }

    /// Depth-first listing, each directory followed by its contents; stops at `MAX_LIST_ENTRIES`.
    fn list_into(&self, dir: &Path, depth: usize, out: &mut Vec<String>) {
        for (path, is_dir) in self.children(dir) {
            if out.len() >= MAX_LIST_ENTRIES {
                if out.last().is_none_or(|l| !l.starts_with("...")) {
                    out.push("... (more entries, list a subdirectory)".to_string());
                }
                return;
            }
            if is_dir {
                out.push(format!("{}/", self.display(&path)));
                if depth > 1 {
                    self.list_into(&path, depth - 1, out);
                }
            } else {
                out.push(self.display(&path));
            }
        }
    }

    fn search_text(&self, query: &str, dir: &str, extension: &str) -> Result<String, String> {
        if query.chars().count() < 2 {
            return Err("query must be at least 2 characters".to_string());
        }
        let needle = query.to_lowercase();
        let extension = extension.trim_start_matches('.').to_lowercase();
        let start = self.resolve(if dir.is_empty() { "." } else { dir })?;

        let mut matches = Vec::new();
        let mut visited = 0usize;
        let mut stack = vec![start];
        'walk: while let Some(path) = stack.pop() {
            if path.is_dir() {
                stack.extend(self.children(&path).into_iter().rev().map(|(p, _)| p));
                continue;
            }
            visited += 1;
            if visited > MAX_FILES_VISITED {
                matches.push("... (search stopped: too many files, narrow it with 'dir')".to_string());
                break;
            }
            if !extension.is_empty()
                && path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(str::to_lowercase)
                    .as_deref()
                    != Some(extension.as_str())
            {
                continue;
            }
            let Some(text) = read_text(&path) else {
                continue;
            };

            for (no, line) in text.lines().enumerate() {
                if line.to_lowercase().contains(&needle) {
                    if matches.len() >= MAX_SEARCH_MATCHES {
                        matches.push("... (more matches, narrow the search)".to_string());
                        break 'walk;
                    }
                    matches.push(format!(
                        "{}:{}: {}",
                        self.display(&path),
                        no + 1,
                        clip(line.trim())
                    ));
                }
            }
        }

        if matches.is_empty() {
            return Ok(format!("no matches for '{query}'"));
        }
        Ok(matches.join("\n"))
    }

    /// Every diagram of the diagrams dir: the model gets the titles, the user a `skw-diagrams` fenced
    /// block with `[{title, url, preview?, previewDark?}]` that the page renders as a gallery.
    fn diagram_gallery(&self) -> Result<(String, String), String> {
        let (dir, url_prefix) = self
            .diagrams
            .as_ref()
            .ok_or("diagrams are not configured")?;
        let mut sources: Vec<PathBuf> = fs::read_dir(dir)
            .map_err(|e| format!("diagrams dir unreadable: {e}"))?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            // `<name>.json` only: archify's receipts are `<name>.<kind>.json`
            .filter(|path| {
                path.extension().is_some_and(|e| e == "json")
                    && path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .is_some_and(|s| !s.contains('.'))
            })
            .collect();
        sources.sort();

        let mut diagrams = Vec::new();
        for source in sources {
            let Some(stem) = source.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let meta = fs::read_to_string(&source)
                .ok()
                .and_then(|text| serde_json::from_str::<Value>(&text).ok())
                .and_then(|json| json.get("meta").cloned())
                .unwrap_or(Value::Null);
            let output = meta
                .get("output")
                .and_then(Value::as_str)
                .map_or_else(|| format!("{stem}.html"), str::to_string);
            // a plain file name inside the dir, so the url can't point elsewhere
            if output.contains(['/', '\\']) || !dir.join(&output).is_file() {
                continue;
            }
            let title = meta.get("title").and_then(Value::as_str).unwrap_or(stem);
            let preview = |theme: &str| {
                let name = format!("{stem}.visual-check.1440x900.{theme}.png");
                dir.join(&name)
                    .is_file()
                    .then(|| format!("{url_prefix}{name}"))
            };

            let mut entry = json!({ "title": title, "url": format!("{url_prefix}{output}") });
            if let Some(light) = preview("light") {
                entry["preview"] = json!(light);
            }
            if let Some(dark) = preview("dark") {
                entry["previewDark"] = json!(dark);
            }
            diagrams.push(entry);
        }

        if diagrams.is_empty() {
            return Err("no diagrams found".to_string());
        }
        let titles: Vec<&str> = diagrams
            .iter()
            .filter_map(|d| d["title"].as_str())
            .collect();
        let note = format!(
            "A gallery of {} diagrams is attached below your answer ({}); a click opens a diagram in full size. \
Reply with one short sentence introducing them, do not list or describe them in detail.",
            diagrams.len(),
            titles.join(", ")
        );
        let shown = format!(
            "```{DIAGRAMS_FENCE_LANG}\n{}\n```",
            serde_json::to_string(&diagrams).map_err(|e| e.to_string())?
        );
        Ok((note, shown))
    }

    /// The model gets a short note, the user the whole file as a fenced Markdown block.
    fn show_file(&self, path: &str) -> Result<(String, String), String> {
        if path.is_empty() {
            return Err("path is required".to_string());
        }
        let resolved = self.resolve(path)?;
        if resolved.is_dir() {
            return Err(format!("'{path}' is a directory, use list_files"));
        }
        let mut text = read_text(&resolved).ok_or_else(|| format!("'{path}' is not a readable text file (binary or bigger than 512 KB)"))?;
        let display = self.display(&resolved);
        let lines = text.lines().count();

        let truncated = text.len() > MAX_SHOWN_CHARS;
        if truncated {
            text.truncate(floor_char_boundary(&text, MAX_SHOWN_CHARS));
        }
        if !text.ends_with('\n') {
            text.push('\n');
        }

        // a fence longer than any backtick run inside, so Markdown files can't close it
        let longest_run = text.split(|c| c != '`').map(str::len).max().unwrap_or(0);
        let fence = "`".repeat(longest_run.max(2) + 1);
        let lang = resolved.extension().and_then(|e| e.to_str()).unwrap_or("");

        let mut shown = format!("**{display}**\n\n{fence}{lang}\n{text}{fence}");
        if truncated {
            shown.push_str(&format!(
                "\n\n(truncated: the file is longer than {MAX_SHOWN_CHARS} characters)"
            ));
        }

        let note = format!(
            "The file {display} ({lines} lines) is attached below your answer, exactly as it is. \
Do not copy it: reply with one short sentence introducing it."
        );
        Ok((note, shown))
    }

    fn read_file(&self, path: &str, start_line: Option<u64>, end_line: Option<u64>) -> Result<String, String> {
        if path.is_empty() {
            return Err("path is required".to_string());
        }
        let resolved = self.resolve(path)?;
        if resolved.is_dir() {
            return Err(format!("'{path}' is a directory, use list_files"));
        }
        let text = read_text(&resolved).ok_or_else(|| format!("'{path}' is not a readable text file (binary or bigger than 512 KB)"))?;

        let total = text.lines().count();
        let start = start_line.unwrap_or(1).max(1) as usize;
        if start > total.max(1) {
            return Err(format!("'{path}' has only {total} lines"));
        }
        let end = end_line
            .map(|e| e as usize)
            .unwrap_or(usize::MAX)
            .min(start + MAX_READ_LINES - 1)
            .min(total)
            .max(start);

        let mut out = format!(
            "{} (lines {start}-{end} of {total})\n",
            self.display(&resolved)
        );
        for (no, line) in text
            .lines()
            .enumerate()
            .skip(start - 1)
            .take(end + 1 - start)
        {
            out.push_str(&format!("{:>5}| {line}\n", no + 1));
        }
        Ok(out)
    }
}

/// `text` in a fenced Markdown block whose fence is longer than any backtick run inside.
fn fenced(text: &str, lang: &str) -> String {
    let longest_run = text.split(|c| c != '`').map(str::len).max().unwrap_or(0);
    let fence = "`".repeat(longest_run.max(2) + 1);
    format!("{fence}{lang}\n{text}\n{fence}")
}

/// File contents if it is small, UTF-8 and has no NUL bytes.
fn read_text(path: &Path) -> Option<String> {
    let meta = fs::metadata(path).ok()?;
    if !meta.is_file() || meta.len() > MAX_FILE_BYTES {
        return None;
    }
    let bytes = fs::read(path).ok()?;
    if bytes.contains(&0) {
        return None;
    }
    String::from_utf8(bytes).ok()
}

fn clip(line: &str) -> String {
    if line.chars().count() <= MAX_LINE_CHARS {
        return line.to_string();
    }
    let mut clipped: String = line.chars().take(MAX_LINE_CHARS).collect();
    clipped.push('…');
    clipped
}

fn floor_char_boundary(s: &str, max: usize) -> usize {
    (0..=max.min(s.len()))
        .rev()
        .find(|&i| s.is_char_boundary(i))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> (PathBuf, CodeSearch) {
        let root = std::env::temp_dir().join(format!("skw-code-search-{}", uuid::Uuid::now_v7()));
        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join("target")).unwrap();
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::write(
            root.join("src/main.rs"),
            "fn main() {\n    println!(\"Hello Needle\");\n}\n",
        )
        .unwrap();
        fs::write(root.join("local.json"), "{\"password\": \"needle\"}").unwrap();
        fs::write(root.join("key.pem"), "needle").unwrap();
        fs::write(root.join("target/out.rs"), "needle").unwrap();
        fs::write(root.join(".git/config"), "needle").unwrap();
        fs::write(root.join("bin.dat"), b"needle\0").unwrap();

        let search = CodeSearch::new(&CodeSearchConfig {
            root: root.clone(),
            max_steps: 3,
            deny_dirs: vec!["target".into()],
            deny_files: vec!["local.json".into()],
            deny_extensions: vec!["pem".into()],
            system_prompt: None,
            diagrams: None,
        });
        (root, search)
    }

    #[test]
    fn search_skips_denied_hidden_and_binary_files() {
        let (root, search) = tree();
        let out = search
            .call("search_text", r#"{"query": "NEEDLE"}"#)
            .for_model;
        assert_eq!(out, "src/main.rs:2: println!(\"Hello Needle\");");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn read_file_refuses_paths_outside_or_denied() {
        let (root, search) = tree();
        for path in [
            "../etc/passwd",
            "/etc/passwd",
            "local.json",
            "key.pem",
            "target/out.rs",
            ".git/config",
            "src/../local.json",
        ] {
            let out = search
                .call("read_file", &json!({ "path": path }).to_string())
                .for_model;
            assert!(out.starts_with("error:"), "{path}: {out}");
        }
        let out = search
            .call(
                "read_file",
                r#"{"path": "./src/main.rs", "start_line": 2, "end_line": 2}"#,
            )
            .for_model;
        assert_eq!(
            out,
            "src/main.rs (lines 2-2 of 3)\n    2|     println!(\"Hello Needle\");\n"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn list_files_hides_denied_entries() {
        let (root, search) = tree();
        assert_eq!(
            search.call("list_files", "{}").for_model,
            "bin.dat\nsrc/\nsrc/main.rs"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn show_file_attaches_the_exact_file() {
        let (root, search) = tree();
        fs::write(root.join("src/doc.md"), "a\n```rust\nx\n```\n").unwrap();

        let out = search.call("show_file", r#"{"path": "src/main.rs"}"#);
        assert!(
            out.for_model.contains("src/main.rs (3 lines)"),
            "{}",
            out.for_model
        );
        assert_eq!(
            out.shown.as_deref(),
            Some("**src/main.rs**\n\n```rs\nfn main() {\n    println!(\"Hello Needle\");\n}\n```")
        );

        let out = search.call("show_file", r#"{"path": "src/doc.md"}"#);
        assert_eq!(
            out.shown.as_deref(),
            Some("**src/doc.md**\n\n````md\na\n```rust\nx\n```\n````")
        );

        let out = search.call("show_file", r#"{"path": "local.json"}"#);
        assert!(out.for_model.starts_with("error:") && out.shown.is_none());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn diagrams_tool_lists_diagram_sources() {
        let (root, _) = tree();
        let dir = root.join("diagrams");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("arch.json"),
            r#"{"meta": {"title": "Arch", "output": "arch.html"}}"#,
        )
        .unwrap();
        fs::write(dir.join("arch.html"), "<html></html>").unwrap();
        fs::write(dir.join("arch.visual-check.1440x900.light.png"), "png").unwrap();
        fs::write(dir.join("arch.finalize.json"), "{}").unwrap();
        fs::write(
            dir.join("evil.json"),
            r#"{"meta": {"output": "../src/main.rs"}}"#,
        )
        .unwrap();

        let search = CodeSearch::new(&CodeSearchConfig {
            root: root.clone(),
            max_steps: 3,
            deny_dirs: vec![],
            deny_files: vec![],
            deny_extensions: vec![],
            system_prompt: None,
            diagrams: Some(DiagramsConfig {
                dir: "diagrams".into(),
                url_prefix: "diagrams/".into(),
            }),
        });
        assert!(search.system_prompt().contains("- diagrams shows"));
        let out = search.call("diagrams", "{}");
        assert!(
            out.for_model.contains("1 diagrams") && out.for_model.contains("Arch"),
            "{}",
            out.for_model
        );
        assert_eq!(
            out.shown.as_deref(),
            Some(
                "```skw-diagrams\n[{\"preview\":\"diagrams/arch.visual-check.1440x900.light.png\",\"title\":\"Arch\",\"url\":\"diagrams/arch.html\"}]\n```"
            )
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn slash_commands_run_tools_directly() {
        let (root, search) = tree();
        assert_eq!(search.run_command("how are you?"), None);
        assert_eq!(search.run_command("/etc/hosts is what?"), None);
        assert_eq!(
            search.run_command("/list_files src"),
            Some("```\nsrc/main.rs\n```".to_string())
        );
        assert_eq!(
            search.run_command("/search_text hello needle ext=rs"),
            Some("```\nsrc/main.rs:2: println!(\"Hello Needle\");\n```".to_string())
        );
        assert_eq!(
            search.run_command("/read_file src/main.rs 2 2"),
            Some("**src/main.rs (lines 2-2 of 3)**\n\n```rs\n    2|     println!(\"Hello Needle\");\n```".to_string())
        );
        assert!(
            search
                .run_command(" /show_file src/main.rs")
                .is_some_and(|a| a.starts_with("**src/main.rs**\n\n```rs\nfn main()"))
        );
        assert_eq!(
            search.run_command("/show_file local.json"),
            Some("`/show_file` failed: 'local.json' is not accessible".to_string())
        );
        // no diagrams configured: /diagrams is unknown, like any other word
        assert!(
            search
                .run_command("/diagrams")
                .is_some_and(|a| a.starts_with("Unknown command `/diagrams`") && !a.contains("- `/diagrams`"))
        );
        assert!(
            search
                .run_command("/help")
                .is_some_and(|a| a.starts_with("Commands run a tool directly"))
        );
        fs::remove_dir_all(root).unwrap();
    }
}

/// Attempts to read outside the root (or denied files inside it) through every tool and every
/// slash command. The fixture plants `SECRET` outside the root, in a sibling dir whose name
/// starts with the root's name, behind symlinks inside the root and in denied/hidden files;
/// none of it may ever reach a tool result.
#[cfg(all(test, unix))]
mod escape_tests {
    use std::os::unix::fs::symlink;

    use super::*;

    const SECRET: &str = "TOP-SECRET-7f3a";

    struct Fixture {
        base: PathBuf,
        root: PathBuf,
        search: CodeSearch,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.base);
        }
    }

    //  base/
    //    outside/secret.txt, outside/nested/secret.rs   (outside the root)
    //    root-evil/secret.txt                           (sibling sharing the root's name prefix)
    //    root/                                          (the root)
    //      src/main.rs                                  (the only readable file with content)
    //      src/link_out.txt  -> base/outside/secret.txt (symlink to a file outside)
    //      src/dir_out       -> base/outside            (symlink to a dir outside)
    //      src/link_abs      -> /etc                    (absolute symlink outside)
    //      src/link_local    -> ../local.json           (symlink inside to a denied file)
    //      src/link_git      -> ../.git                 (symlink inside to a hidden dir)
    //      local.json, key.pem, target/out.rs, .git/config, .env (denied or hidden)
    //      diagrams/evil.json (meta.output pointing outside)
    fn fixture() -> Fixture {
        let base = std::env::temp_dir().join(format!("skw-escape-{}", uuid::Uuid::now_v7()));
        let root = base.join("root");
        for dir in [
            "outside/nested",
            "root-evil",
            "root/src",
            "root/target",
            "root/.git",
            "root/diagrams",
        ] {
            fs::create_dir_all(base.join(dir)).unwrap();
        }
        let secret = format!("password = {SECRET}\n");
        for file in [
            "outside/secret.txt",
            "outside/nested/secret.rs",
            "root-evil/secret.txt",
            "root/local.json",
            "root/key.pem",
            "root/target/out.rs",
            "root/.git/config",
            "root/.env",
        ] {
            fs::write(base.join(file), &secret).unwrap();
        }
        fs::write(root.join("src/main.rs"), "fn main() {}\n").unwrap();
        fs::write(
            root.join("diagrams/evil.json"),
            r#"{"meta": {"title": "Evil", "output": "../../outside/secret.txt"}}"#,
        )
        .unwrap();
        fs::write(
            root.join("diagrams/evil2.json"),
            r#"{"meta": {"title": "Evil2", "output": "/etc/hostname"}}"#,
        )
        .unwrap();
        symlink(
            base.join("outside/secret.txt"),
            root.join("src/link_out.txt"),
        )
        .unwrap();
        symlink(base.join("outside"), root.join("src/dir_out")).unwrap();
        symlink("/etc", root.join("src/link_abs")).unwrap();
        symlink("../local.json", root.join("src/link_local")).unwrap();
        symlink("../.git", root.join("src/link_git")).unwrap();

        let search = CodeSearch::new(&CodeSearchConfig {
            root: root.clone(),
            max_steps: 3,
            deny_dirs: vec!["target".into()],
            deny_files: vec!["local.json".into()],
            deny_extensions: vec!["pem".into()],
            system_prompt: None,
            diagrams: Some(DiagramsConfig {
                dir: "diagrams".into(),
                url_prefix: "diagrams/".into(),
            }),
        });
        Fixture { base, root, search }
    }

    /// Paths that must be refused, as relative, traversal, absolute and symlinked forms.
    fn hostile_paths(f: &Fixture) -> Vec<String> {
        let base = f.base.display().to_string();
        let root = f.root.display().to_string();
        vec![
            "..".into(),
            "../".into(),
            "../outside/secret.txt".into(),
            "../../../../../../etc/passwd".into(),
            "./../outside/secret.txt".into(),
            "src/../../outside/secret.txt".into(),
            "src/../../root-evil/secret.txt".into(),
            "../root-evil/secret.txt".into(),
            "/etc/passwd".into(),
            "/etc".into(),
            "/".into(),
            format!("{base}/outside/secret.txt"),
            format!("{base}/outside"),
            format!("{root}-evil/secret.txt"),
            format!("{root}/../outside/secret.txt"),
            format!("{root}/local.json"),
            "src/link_out.txt".into(),
            "src/dir_out".into(),
            "src/dir_out/secret.txt".into(),
            "src/dir_out/nested/secret.rs".into(),
            "src/link_abs".into(),
            "src/link_abs/passwd".into(),
            "src/link_local".into(),
            "src/link_git".into(),
            "src/link_git/config".into(),
            "local.json".into(),
            "./local.json".into(),
            "src/../local.json".into(),
            "key.pem".into(),
            "KEY.PEM".into(),
            "target/out.rs".into(),
            "target".into(),
            ".git/config".into(),
            ".git".into(),
            ".env".into(),
            "src/../.git/config".into(),
            "..\\outside\\secret.txt".into(),
            "%2e%2e/outside/secret.txt".into(),
            "~/.ssh/id_ed25519".into(),
            "src/main.rs\0../../outside/secret.txt".into(),
        ]
    }

    #[track_caller]
    fn assert_refused(f: &Fixture, tool: &str, args: Value) {
        let out = f.search.call(tool, &args.to_string());
        assert!(
            out.for_model.starts_with("error:"),
            "{tool} {args} was not refused: {}",
            out.for_model
        );
        assert!(out.shown.is_none(), "{tool} {args} attached something");
        assert!(
            !out.for_model.contains(SECRET),
            "{tool} {args} leaked the secret"
        );
    }

    #[test]
    fn read_and_show_file_refuse_every_hostile_path() {
        let f = fixture();
        for path in hostile_paths(&f) {
            assert_refused(&f, "read_file", json!({ "path": path }));
            assert_refused(
                &f,
                "read_file",
                json!({ "path": path, "start_line": 1, "end_line": 5 }),
            );
            assert_refused(&f, "show_file", json!({ "path": path }));
        }
    }

    #[test]
    fn list_files_and_search_refuse_hostile_dirs() {
        let f = fixture();
        for dir in hostile_paths(&f) {
            // ".." style paths that resolve to the root's parent, or outside, or a denied dir
            assert_refused(&f, "list_files", json!({ "dir": dir, "depth": 4 }));
            assert_refused(
                &f,
                "search_text",
                json!({ "query": "password", "dir": dir }),
            );
        }
    }

    #[test]
    fn whole_tree_listing_and_search_never_follow_symlinks_or_show_denied_files() {
        let f = fixture();
        let listing = f.search.call("list_files", r#"{"depth": 4}"#).for_model;
        assert_eq!(
            listing,
            "diagrams/\ndiagrams/evil.json\ndiagrams/evil2.json\nsrc/\nsrc/main.rs"
        );

        for query in [SECRET, "password"] {
            let out = f
                .search
                .call("search_text", &json!({ "query": query }).to_string());
            assert_eq!(out.for_model, format!("no matches for '{query}'"));
        }
    }

    #[test]
    fn diagrams_refuse_outputs_outside_their_dir() {
        let f = fixture();
        let out = f.search.call("diagrams", "{}");
        assert_eq!(out.for_model, "error: no diagrams found");
        assert!(out.shown.is_none());
    }

    #[test]
    fn slash_commands_refuse_every_hostile_path() {
        let f = fixture();
        for path in hostile_paths(&f) {
            // a NUL byte can't be typed into a command word list meaningfully; whitespace splits it
            for command in [
                format!("/read_file {path}"),
                format!("/show_file {path}"),
                format!("/list_files {path} 4"),
            ] {
                let answer = f.search.run_command(&command).unwrap_or_default();
                assert!(!answer.contains(SECRET), "{command:?} leaked the secret");
                assert!(
                    answer.contains("failed:"),
                    "{command:?} was not refused: {answer}"
                );
            }
            let answer = f
                .search
                .run_command(&format!("/search_text password dir={path}"))
                .unwrap_or_default();
            assert!(
                !answer.contains(SECRET),
                "/search_text dir={path} leaked the secret"
            );
        }
    }

    #[test]
    fn paths_inside_the_root_still_work() {
        let f = fixture();
        let root = f.root.display().to_string();
        for path in [
            "src/main.rs",
            "./src/main.rs",
            "src/../src/main.rs",
            "src/./main.rs",
            &format!("{root}/src/main.rs"),
        ] {
            let out = f
                .search
                .call("read_file", &json!({ "path": path }).to_string());
            assert!(
                out.for_model.starts_with("src/main.rs (lines 1-1 of 1)"),
                "{path}: {}",
                out.for_model
            );
        }
        assert_eq!(
            f.search
                .call("list_files", r#"{"dir": "src/.."}"#)
                .for_model,
            "diagrams/\ndiagrams/evil.json\ndiagrams/evil2.json\nsrc/\nsrc/main.rs"
        );
    }
}
