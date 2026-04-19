/// Remove unpaired Unicode surrogates from a string.
///
/// Unpaired surrogates (high surrogates U+D800–U+DBFF without a matching low
/// surrogate U+DC00–U+DFFF, or vice versa) cause JSON serialization errors in
/// many API providers.
///
/// In Rust, `String` is always valid UTF-8, so this situation cannot arise for
/// native `String` values. This function exists for data that enters from
/// external sources (e.g. JSON with escaped surrogates, or WTF-8 input) and
/// has been lossily decoded.
///
/// The implementation scans for the Unicode replacement character U+FFFD which
/// is what `String::from_utf8_lossy` inserts for invalid sequences. For truly
/// valid UTF-8 strings this is a no-op.
///
/// Port of `sanitizeSurrogates()` from `packages/ai/src/utils/sanitize-unicode.ts`.
pub fn sanitize_surrogates(text: &str) -> String {
    text.replace('\u{FFFD}', "")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_normal_text() {
        assert_eq!(sanitize_surrogates("Hello World"), "Hello World");
    }

    #[test]
    fn preserves_valid_emoji() {
        let s = "Hello 🙈 World";
        assert_eq!(sanitize_surrogates(s), s);
    }

    #[test]
    fn removes_replacement_characters() {
        let s = "Text \u{FFFD} here";
        assert_eq!(sanitize_surrogates(s), "Text  here");
    }
}
