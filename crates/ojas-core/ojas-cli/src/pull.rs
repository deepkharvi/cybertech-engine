//! Model manager for The Cybertech Engine (Cybertech addition, not part of upstream Ojas).
//!
//! `cybertech-engine pull qwen2.5:7b`   download a GGUF model from Hugging Face
//! `cybertech-engine list`              show downloaded models
//! `cybertech-engine rm qwen2.5:7b`     delete a downloaded model
//! `cybertech-engine serve qwen2.5:7b`  any command that takes a model also accepts a pulled name
//!
//! Downloads are done with the system `curl` (built into Windows 10/11, macOS and Linux), so
//! this adds no new Rust dependencies. Interrupted downloads resume when `pull` is re-run.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_QUANT: &str = "Q4_K_M";

/// Short names -> Hugging Face GGUF repositories. Any `owner/repo` also works.
const ALIASES: &[(&str, &str)] = &[
    ("qwen2.5:3b", "Qwen/Qwen2.5-3B-Instruct-GGUF"),
    ("qwen2.5:7b", "Qwen/Qwen2.5-7B-Instruct-GGUF"),
    ("qwen2.5-coder:7b", "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"),
    ("llama3.2:3b", "bartowski/Llama-3.2-3B-Instruct-GGUF"),
    ("llama3.1:8b", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"),
];

/// Where models are stored. Override with the CYBERTECH_MODELS environment variable.
pub fn models_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("CYBERTECH_MODELS") {
        return PathBuf::from(d);
    }
    #[cfg(windows)]
    {
        if let Some(d) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(d).join("CybertechEngine").join("models");
        }
    }
    if let Some(h) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        return PathBuf::from(h).join(".cybertech-engine").join("models");
    }
    PathBuf::from("models_gguf")
}

/// Folder-safe version of a model name: `qwen2.5:7b` -> `qwen2.5-7b`.
fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' { c } else { '-' })
        .collect()
}

fn dir_name(name: &str, quant: &str) -> String {
    if quant.eq_ignore_ascii_case(DEFAULT_QUANT) {
        sanitize(name)
    } else {
        format!("{}-{}", sanitize(name), sanitize(quant))
    }
}

/// Returns (huggingface repo, quantization) for a pull target.
fn parse_target(name: &str, quant_arg: Option<&str>) -> Result<(String, String)> {
    let mut n = name.trim();
    for prefix in ["hf.co/", "huggingface.co/", "https://huggingface.co/"] {
        if let Some(rest) = n.strip_prefix(prefix) {
            n = rest;
        }
    }
    if let Some((_, repo)) = ALIASES.iter().find(|(a, _)| a.eq_ignore_ascii_case(n)) {
        let q = quant_arg.unwrap_or(DEFAULT_QUANT);
        return Ok((repo.to_string(), q.to_string()));
    }
    if n.contains('/') {
        // owner/repo or owner/repo:QUANT
        if let Some((repo, q)) = n.rsplit_once(':') {
            return Ok((repo.to_string(), quant_arg.unwrap_or(q).to_string()));
        }
        return Ok((n.to_string(), quant_arg.unwrap_or(DEFAULT_QUANT).to_string()));
    }
    let known: Vec<&str> = ALIASES.iter().map(|(a, _)| *a).collect();
    bail!(
        "unknown model {name:?}. Built-in names: {}. Or use a Hugging Face repo: owner/repo",
        known.join(", ")
    )
}

fn curl_text(url: &str) -> Result<String> {
    let out = Command::new("curl")
        .args(["-sSL", "--fail", url])
        .output()
        .context("could not run `curl` (it ships with Windows 10/11; is it on PATH?)")?;
    if !out.status.success() {
        bail!(
            "download lookup failed for {url}: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn gguf_files_from_json(json: &str) -> Result<Vec<String>> {
    let v: serde_json::Value = serde_json::from_str(json).context("unexpected reply from Hugging Face")?;
    let siblings = v
        .get("siblings")
        .and_then(|s| s.as_array())
        .context("Hugging Face reply has no file list (is the repository name right?)")?;
    Ok(siblings
        .iter()
        .filter_map(|s| s.get("rfilename").and_then(|f| f.as_str()))
        .filter(|f| f.to_ascii_lowercase().ends_with(".gguf"))
        .map(str::to_string)
        .collect())
}

/// Choose the file(s) for one quantization. Split models return every shard.
fn pick(files: &[String], quant: &str) -> Result<Vec<String>> {
    let q = quant.to_ascii_lowercase();
    let mut m: Vec<String> = files
        .iter()
        .filter(|f| {
            let l = f.to_ascii_lowercase();
            l.contains(&q) && !l.contains("mmproj") && !l.contains("imatrix")
        })
        .cloned()
        .collect();
    if m.is_empty() {
        let sample: Vec<&str> = files.iter().take(15).map(String::as_str).collect();
        bail!(
            "no {quant} file in that repository. Files found: {}",
            if sample.is_empty() { "(none)".to_string() } else { sample.join(", ") }
        );
    }
    m.sort();
    if m.iter().any(|f| f.contains("-of-")) {
        m.retain(|f| f.contains("-of-"));
    } else {
        m.truncate(1);
    }
    Ok(m)
}

fn download(url: &str, dest: &Path) -> Result<()> {
    if let Ok(md) = std::fs::metadata(dest) {
        if md.len() > 0 {
            println!("already downloaded: {}", dest.display());
            return Ok(());
        }
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let part = PathBuf::from(format!("{}.part", dest.display()));
    println!("downloading {}", dest.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
    let status = Command::new("curl")
        .args(["-L", "--fail", "-C", "-", "--progress-bar", "-o"])
        .arg(&part)
        .arg(url)
        .status()
        .context("could not run `curl` (it ships with Windows 10/11; is it on PATH?)")?;
    if !status.success() {
        bail!("download failed (re-run the same command to resume): {url}");
    }
    std::fs::rename(&part, dest)?;
    Ok(())
}

pub fn pull(name: &str, quant_arg: Option<&str>) -> Result<()> {
    let (repo, quant) = parse_target(name, quant_arg)?;
    println!("Looking up {repo} ({quant}) ...");
    let json = curl_text(&format!("https://huggingface.co/api/models/{repo}"))?;
    let chosen = pick(&gguf_files_from_json(&json)?, &quant)?;
    let dir = models_dir().join(dir_name(name, &quant));
    for f in &chosen {
        let base = Path::new(f).file_name().context("bad file name")?;
        download(&format!("https://huggingface.co/{repo}/resolve/main/{f}"), &dir.join(base))?;
    }
    println!("Done. Saved in {}", dir.display());
    println!("Run it:  cybertech-engine serve {name}");
    Ok(())
}

fn gguf_in(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map(|x| x.eq_ignore_ascii_case("gguf")).unwrap_or(false))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

fn first_gguf(dir: &Path) -> Option<PathBuf> {
    let files = gguf_in(dir);
    files
        .iter()
        .find(|p| p.to_string_lossy().contains("-00001-of-"))
        .cloned()
        .or_else(|| files.into_iter().next())
}

pub fn list() -> Result<()> {
    let root = models_dir();
    let mut rows: Vec<(String, u64, usize)> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&root) {
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if !p.is_dir() {
                continue;
            }
            let files = gguf_in(&p);
            if files.is_empty() {
                continue;
            }
            let size: u64 = files.iter().filter_map(|f| std::fs::metadata(f).ok()).map(|m| m.len()).sum();
            rows.push((e.file_name().to_string_lossy().into_owned(), size, files.len()));
        }
    }
    rows.sort();
    if rows.is_empty() {
        println!("No models yet. Try:  cybertech-engine pull qwen2.5:3b");
        println!("(models are stored in {})", root.display());
        return Ok(());
    }
    println!("{:<44} {:>10}  {}", "NAME", "SIZE", "FILES");
    for (n, s, c) in rows {
        println!("{:<44} {:>7.1} GB  {}", n, s as f64 / 1e9, c);
    }
    println!("\nstored in {}", root.display());
    Ok(())
}

pub fn remove(name: &str) -> Result<()> {
    let root = models_dir();
    for cand in [sanitize(name), name.to_string()] {
        let d = root.join(&cand);
        if d.is_dir() && !cand.is_empty() && !cand.contains("..") {
            std::fs::remove_dir_all(&d)?;
            println!("deleted {}", d.display());
            return Ok(());
        }
    }
    bail!("no downloaded model named {name:?} (see `cybertech-engine list`)")
}

fn resolve_in(root: &Path, arg: &str) -> String {
    if Path::new(arg).exists() {
        return arg.to_string();
    }
    for cand in [sanitize(arg), arg.to_string()] {
        if cand.is_empty() || cand.contains("..") {
            continue;
        }
        let d = root.join(&cand);
        if d.is_dir() {
            if let Some(f) = first_gguf(&d) {
                return f.to_string_lossy().into_owned();
            }
        }
    }
    arg.to_string()
}

/// A path to a model file, or the name of a pulled model -> the path of its (first) file.
/// Anything not found is returned unchanged so the normal "file not found" error still shows.
pub fn resolve(arg: &str) -> String {
    resolve_in(&models_dir(), arg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizes_names() {
        assert_eq!(sanitize("qwen2.5:7b"), "qwen2.5-7b");
        assert_eq!(sanitize("Qwen/Qwen2.5-7B-Instruct-GGUF"), "Qwen-Qwen2.5-7B-Instruct-GGUF");
        assert_eq!(dir_name("qwen2.5:7b", "Q4_K_M"), "qwen2.5-7b");
        assert_eq!(dir_name("qwen2.5:7b", "Q8_0"), "qwen2.5-7b-Q8_0");
    }

    #[test]
    fn parses_targets() {
        assert_eq!(parse_target("qwen2.5:7b", None).unwrap().0, "Qwen/Qwen2.5-7B-Instruct-GGUF");
        assert_eq!(parse_target("qwen2.5:7b", Some("Q8_0")).unwrap().1, "Q8_0");
        assert_eq!(parse_target("hf.co/foo/bar:Q5_K_M", None).unwrap(), ("foo/bar".into(), "Q5_K_M".into()));
        assert_eq!(parse_target("foo/bar", None).unwrap(), ("foo/bar".into(), "Q4_K_M".into()));
        assert!(parse_target("nonsense", None).is_err());
    }

    #[test]
    fn reads_hf_json_and_picks_shards() {
        let json = r#"{"siblings":[{"rfilename":"README.md"},
          {"rfilename":"qwen-q4_k_m-00002-of-00002.gguf"},
          {"rfilename":"qwen-q4_k_m-00001-of-00002.gguf"},
          {"rfilename":"qwen-q8_0.gguf"},{"rfilename":"mmproj-q4_k_m.gguf"}]}"#;
        let files = gguf_files_from_json(json).unwrap();
        let p = pick(&files, "Q4_K_M").unwrap();
        assert_eq!(p, vec!["qwen-q4_k_m-00001-of-00002.gguf", "qwen-q4_k_m-00002-of-00002.gguf"]);
        assert_eq!(pick(&files, "q8_0").unwrap(), vec!["qwen-q8_0.gguf"]);
        assert!(pick(&files, "Q2_K").is_err());
    }

    #[test]
    fn resolves_pulled_names() {
        let root = std::env::temp_dir().join(format!("cybertech-test-{}", std::process::id()));
        let d = root.join("qwen2.5-7b");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("m-00002-of-00002.gguf"), b"x").unwrap();
        std::fs::write(d.join("m-00001-of-00002.gguf"), b"x").unwrap();
        let r = resolve_in(&root, "qwen2.5:7b");
        assert!(r.ends_with("m-00001-of-00002.gguf"), "{r}");
        assert_eq!(resolve_in(&root, "nothing-here"), "nothing-here");
        std::fs::remove_dir_all(&root).ok();
    }
}
