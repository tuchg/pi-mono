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

    // Try progressively closing open brackets/braces.
    let mut buf = input.to_string();
    let mut open_braces = 0i32;
    let mut open_brackets = 0i32;
    let mut in_string = false;
    let mut escape = false;

    for ch in buf.chars() {
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

    // Close any open string
    if in_string {
        buf.push('"');
    }

    // Close open brackets and braces
    for _ in 0..open_brackets {
        buf.push(']');
    }
    for _ in 0..open_braces {
        buf.push('}');
    }

    serde_json::from_str::<serde_json::Value>(&buf).ok()
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
}
