//! The site owner's resume for the system prompt (`/llm/profile`), so the assistant can answer
//! questions about them. The source is a resume page (HTML, reduced to plain text) or a text file,
//! read once at startup.

use std::{fs, path::PathBuf};

use serde_derive::Deserialize;

/// `/llm/profile` config section.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProfileConfig {
    /// Resume file, relative to the CWD (crate dir); `.html`/`.htm` is reduced to text.
    pub file: PathBuf,
    /// Whose resume it is, e.g. "Sergey Kutsenko".
    pub name: String,
    /// Only the part of the file before this marker is used (e.g. the start of the page's chat widget).
    #[serde(default)]
    pub end_marker: Option<String>,
    /// System prompt section; `{name}` and `{resume}` are replaced.
    #[serde(default)]
    pub prompt: Option<String>,
}

const DEFAULT_PROMPT: &str = "This chat runs on the resume website of {name}. Answer questions about {name} \
(experience, jobs, projects, skills, education, contacts, availability) from the resume below, without calling tools: \
the code tools know nothing about {name}. Do not invent facts that are not in the resume; if it does not mention \
something, say so. Jobs are listed with year ranges: a year inside a range (e.g. 2024 in 2023–2025) is a year he \
worked there. Speak about {name} in the third person.\n\n\
Resume of {name}:\n{resume}";

/// The system prompt section with the resume; panics if the file can't be read (startup only).
pub fn load_prompt(cfg: &ProfileConfig) -> String {
    let raw = fs::read_to_string(&cfg.file).unwrap_or_else(|e| panic!("/llm/profile/file {:?}: {e}", cfg.file));
    let raw = match &cfg.end_marker {
        Some(marker) => {
            raw.split_once(marker.as_str())
                .unwrap_or_else(|| {
                    panic!(
                        "/llm/profile/endMarker {marker:?} not found in {:?}",
                        cfg.file
                    )
                })
                .0
        }
        None => raw.as_str(),
    };

    let is_html = cfg
        .file
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("html") || e.eq_ignore_ascii_case("htm"));
    let resume = if is_html {
        html_to_text(raw)
    } else {
        raw.trim().to_string()
    };

    cfg.prompt
        .as_deref()
        .unwrap_or(DEFAULT_PROMPT)
        .replace("{name}", &cfg.name)
        .replace("{resume}", &resume)
}

/// Visible text of an HTML document: `<head>`, scripts, styles and svgs are dropped, block elements
/// become line breaks, list items `- ` lines; whitespace is collapsed and blank lines removed.
fn html_to_text(html: &str) -> String {
    let html = match find_ci(html, "<body") {
        Some(i) => &html[i..],
        None => html,
    };

    let mut out = String::with_capacity(html.len() / 2);
    let mut rest = html;
    while let Some(lt) = rest.find('<') {
        out.push_str(&decode_entities(&rest[..lt]));
        let Some(gt) = rest[lt..].find('>') else {
            rest = "";
            break;
        };
        let tag = &rest[lt + 1..lt + gt];
        rest = &rest[lt + gt + 1..];

        let name: String = tag
            .trim_start_matches('/')
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();
        let closing = tag.starts_with('/');

        if !closing
            && matches!(
                name.as_str(),
                "script" | "style" | "svg" | "head" | "template"
            )
        {
            // skip the element with its contents
            match find_ci(rest, &format!("</{name}")) {
                Some(end) => {
                    rest = &rest[end..];
                    rest = rest.find('>').map_or("", |gt| &rest[gt + 1..]);
                }
                None => rest = "",
            }
            continue;
        }

        out.push_str(match name.as_str() {
            "li" if !closing => "\n- ",
            "p" | "div" | "section" | "header" | "footer" | "main" | "article" | "ul" | "ol" | "li" | "br" | "tr" | "h1" | "h2" | "h3" | "h4"
            | "h5" | "h6" => "\n",
            // inline elements still separate words (e.g. skill chips without whitespace between them)
            _ => " ",
        });
    }
    out.push_str(&decode_entities(rest));

    out.lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty() && line != "-")
        .collect::<Vec<_>>()
        .join("\n")
}

fn find_ci(haystack: &str, needle: &str) -> Option<usize> {
    haystack
        .to_ascii_lowercase()
        .find(&needle.to_ascii_lowercase())
}

fn decode_entities(text: &str) -> String {
    if !text.contains('&') {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let after = &rest[amp + 1..];
        let decoded = after.find(';').filter(|&semi| semi <= 10).and_then(|semi| {
            let entity = &after[..semi];
            let ch = match entity {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                "nbsp" => Some(' '),
                _ if entity.starts_with("#x") || entity.starts_with("#X") => u32::from_str_radix(&entity[2..], 16)
                    .ok()
                    .and_then(char::from_u32),
                _ if entity.starts_with('#') => entity[1..].parse().ok().and_then(char::from_u32),
                _ => None,
            };
            ch.map(|c| (c, semi))
        });
        match decoded {
            Some((c, semi)) => {
                out.push(c);
                rest = &after[semi + 1..];
            }
            None => {
                out.push('&');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_is_reduced_to_visible_text() {
        let html = r#"<html><head><title>T</title><style>p{}</style></head><body>
<h1>Jane&nbsp;Doe</h1><p>Rust &amp; SQL &lt;3</p>
<script>var x = "<p>no</p>";</script><svg><text>no</text></svg>
<ul><li>One</li><li><strong>Two</strong></li></ul><span>A</span><span>B</span>&#x2192;&#8212;&bogus;</body></html>"#;
        assert_eq!(
            html_to_text(html),
            "Jane Doe\nRust & SQL <3\n- One\n- Two\nA B →—&bogus;"
        );
    }

    #[test]
    fn prompt_uses_the_part_before_the_end_marker() {
        let file = std::env::temp_dir().join(format!("skw-profile-{}.html", uuid::Uuid::now_v7()));
        fs::write(
            &file,
            "<body><p>Senior dev</p><div id=chat><p>chat help</p></div></body>",
        )
        .unwrap();
        let prompt = load_prompt(&ProfileConfig {
            file: file.clone(),
            name: "Jane".into(),
            end_marker: Some("<div id=chat>".into()),
            prompt: Some("{name}: {resume}".into()),
        });
        assert_eq!(prompt, "Jane: Senior dev");
        fs::remove_file(file).unwrap();
    }
}
