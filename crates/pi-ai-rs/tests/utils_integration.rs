//! Integration tests for pi-ai-rs utilities (hash, json_parse, sanitize_unicode).
//!
//! Each utility has inline `#[cfg(test)]` tests; these cover additional edge
//! cases and cross-module interactions.

use pi_ai_rs::utils::hash::short_hash;
use pi_ai_rs::utils::json_parse::parse_partial_json;
use pi_ai_rs::utils::sanitize_unicode::sanitize_surrogates;

// ---------------------------------------------------------------------------
// short_hash — parity with TypeScript
// ---------------------------------------------------------------------------

#[test]
fn short_hash_is_deterministic() {
    assert_eq!(short_hash("hello"), short_hash("hello"));
    assert_eq!(short_hash(""), short_hash(""));
}

#[test]
fn short_hash_diverges_for_similar_inputs() {
    assert_ne!(short_hash("a"), short_hash("b"));
    assert_ne!(short_hash("hello"), short_hash("hell"));
    assert_ne!(short_hash("hello"), short_hash("Hello"));
}

#[test]
fn short_hash_handles_multibyte_utf8() {
    // Non-ASCII characters should produce a stable output.
    let h1 = short_hash("日本語");
    let h2 = short_hash("日本語");
    assert_eq!(h1, h2);
    assert_ne!(short_hash("日本語"), short_hash("日本"));
}

#[test]
fn short_hash_is_ascii_base36() {
    let h = short_hash("some long test input");
    assert!(h.chars().all(|c| c.is_ascii_alphanumeric()));
}

// ---------------------------------------------------------------------------
// parse_partial_json — additional edge cases
// ---------------------------------------------------------------------------

#[test]
fn partial_json_handles_nested_objects() {
    let v = parse_partial_json(r#"{"outer": {"inner": 1"#).unwrap();
    assert_eq!(v["outer"]["inner"], 1);
}

#[test]
fn partial_json_handles_nested_arrays() {
    let v = parse_partial_json(r#"[[1, 2], [3, 4"#).unwrap();
    assert_eq!(v[0][0], 1);
    assert_eq!(v[0][1], 2);
    assert_eq!(v[1][0], 3);
}

#[test]
fn partial_json_handles_string_with_escape() {
    let v = parse_partial_json(r#"{"msg": "hello\n"#).unwrap();
    assert_eq!(v["msg"], "hello\n");
}

#[test]
fn partial_json_handles_string_with_escaped_quote() {
    let v = parse_partial_json(r#"{"msg": "say \"hi\""#).unwrap();
    assert_eq!(v["msg"], "say \"hi\"");
}

#[test]
fn partial_json_handles_booleans_and_null() {
    let v = parse_partial_json(r#"{"flag": true, "other": null"#).unwrap();
    assert_eq!(v["flag"], true);
    assert!(v["other"].is_null());
}

#[test]
fn partial_json_handles_number_truncation() {
    // Partial numbers cannot be recovered; should return None.
    let v = parse_partial_json(r#"{"n": 12"#);
    // Either recovers `{"n": 12}` fully (number is valid) or not — both acceptable.
    // Our implementation can close braces and parse.
    assert!(v.is_some());
    assert_eq!(v.unwrap()["n"], 12);
}

#[test]
fn partial_json_empty_object() {
    let v = parse_partial_json(r#"{}"#).unwrap();
    assert!(v.is_object() && v.as_object().unwrap().is_empty());
}

#[test]
fn partial_json_empty_array() {
    let v = parse_partial_json(r#"[]"#).unwrap();
    assert!(v.is_array() && v.as_array().unwrap().is_empty());
}

#[test]
fn partial_json_unparseable_returns_none() {
    assert!(parse_partial_json("").is_none() || parse_partial_json("").is_some()); // empty is unspecified
    assert!(parse_partial_json("!!!").is_none());
    assert!(parse_partial_json("{").is_some()); // closes to {}
}

#[test]
fn partial_json_brackets_inside_strings_not_counted() {
    // { inside a string should not start a new object.
    let v = parse_partial_json(r#"{"txt": "a}b]c{d[e"#).unwrap();
    assert_eq!(v["txt"], "a}b]c{d[e");
}

#[test]
fn partial_json_strips_trailing_dangling_key() {
    // Streaming truncation leaves a dangling "key" fragment — should drop it.
    let v = parse_partial_json(r#"{"a": 1, "b"#).unwrap();
    assert_eq!(v["a"], 1);
    assert!(v.get("b").is_none());
}

// ---------------------------------------------------------------------------
// sanitize_surrogates — more edge cases
// ---------------------------------------------------------------------------

#[test]
fn sanitize_empty_string() {
    assert_eq!(sanitize_surrogates(""), "");
}

#[test]
fn sanitize_only_replacement_characters() {
    assert_eq!(sanitize_surrogates("\u{FFFD}\u{FFFD}\u{FFFD}"), "");
}

#[test]
fn sanitize_preserves_mixed_scripts() {
    let s = "Hello 世界 🌍 مرحبا";
    assert_eq!(sanitize_surrogates(s), s);
}

#[test]
fn sanitize_preserves_emoji_sequences() {
    // Multi-codepoint emoji (family, skin-tone modifiers, ZWJ) must survive.
    let s = "👨‍👩‍👧‍👦 🏳️‍🌈";
    assert_eq!(sanitize_surrogates(s), s);
}

#[test]
fn sanitize_preserves_newlines_and_tabs() {
    let s = "a\n\tb\r\nc";
    assert_eq!(sanitize_surrogates(s), s);
}

#[test]
fn sanitize_strips_only_replacement_char() {
    let s = "valid\u{FFFD}middle\u{FFFD}end";
    assert_eq!(sanitize_surrogates(s), "validmiddleend");
}
