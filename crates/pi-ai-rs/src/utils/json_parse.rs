/// Attempt to parse a partial JSON string that may be truncated mid-value.
///
/// This is useful for streaming tool call arguments where the JSON arrives
/// incrementally. Returns `Some(value)` if the input can be salvaged into
/// valid JSON, otherwise `None`.
pub fn parse_partial_json(input: &str) -> Option<serde_json::Value> {
    // Fast path: try normal parse first.
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(input) {
        return Some(v);
    }

    // Scan the input to track structural state.
    let mut open_braces = 0i32;
    let mut open_brackets = 0i32;
    let mut in_string = false;
    let mut escape = false;

    for ch in input.chars() {
        if escape {
            escape = false;
            continue;
        }
        match ch {
            '\\' if in_string => escape = true,
            '"' => in_string = !in_string,
            '{' if !in_string => open_braces += 1,
            '}' if !in_string => open_braces -= 1,
            '[' if !in_string => open_brackets += 1,
            ']' if !in_string => open_brackets -= 1,
            _ => {}
        }
    }

    let mut buf = input.to_string();

    // Close any open string.
    if in_string {
        buf.push('"');
    }

    // Strip trailing commas and colons (common streaming truncation points).
    loop {
        let trimmed = buf.trim_end();
        if trimmed.ends_with(',') || trimmed.ends_with(':') {
            let new_len = trimmed.len() - 1;
            buf.truncate(new_len);
        } else {
            break;
        }
    }

    // Attempt 1: close open brackets and braces.
    let mut attempt = buf.clone();
    close_containers(&mut attempt, open_brackets, open_braces);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&attempt) {
        return Some(v);
    }

    // Attempt 2: the remaining content may still have a dangling key (e.g.
    // `{"a": 1, "b"` after stripping the colon). Strip back to the last comma
    // which removes the incomplete key-value pair.
    if let Some(comma_pos) = buf.rfind(',') {
        let mut attempt2 = buf[..comma_pos].to_string();
        close_containers(&mut attempt2, open_brackets, open_braces);
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&attempt2) {
            return Some(v);
        }
    }

    None
}

fn close_containers(buf: &mut String, open_brackets: i32, open_braces: i32) {
    for _ in 0..open_brackets {
        buf.push(']');
    }
    for _ in 0..open_braces {
        buf.push('}');
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_json_parses() {
        let v = parse_partial_json(r#"{"a": 1}"#);
        assert!(v.is_some());
        assert_eq!(v.unwrap()["a"], 1);
    }

    #[test]
    fn partial_object_closes() {
        let v = parse_partial_json(r#"{"a": 1"#);
        assert!(v.is_some());
    }

    #[test]
    fn partial_array_closes() {
        let v = parse_partial_json(r#"[1, 2"#);
        assert!(v.is_some());
    }

    #[test]
    fn nonsense_returns_none() {
        let v = parse_partial_json("not json at all");
        assert!(v.is_none());
    }

    #[test]
    fn trailing_comma_in_object() {
        let v = parse_partial_json(r#"{"a": 1,"#);
        assert!(v.is_some());
        assert_eq!(v.unwrap()["a"], 1);
    }

    #[test]
    fn trailing_colon_in_object() {
        let v = parse_partial_json(r#"{"a": 1, "b":"#);
        assert!(v.is_some());
        assert_eq!(v.unwrap()["a"], 1);
    }

    #[test]
    fn trailing_comma_in_array() {
        let v = parse_partial_json(r#"[1, 2,"#);
        assert!(v.is_some());
        let arr = v.unwrap();
        assert_eq!(arr[0], 1);
        assert_eq!(arr[1], 2);
    }

    #[test]
    fn partial_string_value() {
        let v = parse_partial_json(r#"{"path": "/foo/ba"#);
        assert!(v.is_some());
        assert_eq!(v.unwrap()["path"], "/foo/ba");
    }
}
