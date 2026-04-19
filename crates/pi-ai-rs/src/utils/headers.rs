use std::collections::HashMap;

/// Convert an HTTP `HeaderMap` into a `HashMap<String, String>`.
///
/// Port of `headersToRecord()` from `packages/ai/src/utils/headers.ts`.
pub fn headers_to_map(headers: &reqwest::header::HeaderMap) -> HashMap<String, String> {
    let mut result = HashMap::new();
    for (name, value) in headers.iter() {
        if let Ok(v) = value.to_str() {
            result.insert(name.as_str().to_string(), v.to_string());
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderValue};

    #[test]
    fn converts_headers() {
        let mut hm = HeaderMap::new();
        hm.insert("content-type", HeaderValue::from_static("application/json"));
        hm.insert("x-custom", HeaderValue::from_static("value"));

        let map = headers_to_map(&hm);
        assert_eq!(map.get("content-type").unwrap(), "application/json");
        assert_eq!(map.get("x-custom").unwrap(), "value");
    }
}
