/// Fast deterministic hash to shorten long strings.
///
/// Port of the TypeScript `shortHash()` from `packages/ai/src/utils/hash.ts`.
/// Uses a pair of multiply-xorshift hashes (FNV-like) and encodes the two
/// 32-bit halves as base-36 strings.
pub fn short_hash(s: &str) -> String {
    let mut h1: u32 = 0xdeadbeef;
    let mut h2: u32 = 0x41c6ce57;

    for ch in s.encode_utf16() {
        let c = ch as u32;
        h1 = (h1 ^ c).wrapping_mul(2654435761);
        h2 = (h2 ^ c).wrapping_mul(1597334677);
    }

    h1 = (h1 ^ (h1 >> 16)).wrapping_mul(2246822507) ^ (h2 ^ (h2 >> 13)).wrapping_mul(3266489909);
    h2 = (h2 ^ (h2 >> 16)).wrapping_mul(2246822507) ^ (h1 ^ (h1 >> 13)).wrapping_mul(3266489909);

    format!("{}{}", radix_36(h2), radix_36(h1))
}

/// Encode a `u32` as a base-36 string (digits 0-9 then a-z), matching
/// JavaScript's `(n >>> 0).toString(36)`.
fn radix_36(n: u32) -> String {
    if n == 0 {
        return "0".to_string();
    }
    let mut v = n;
    let mut buf = Vec::new();
    while v > 0 {
        let digit = (v % 36) as u8;
        buf.push(if digit < 10 { b'0' + digit } else { b'a' + digit - 10 });
        v /= 36;
    }
    buf.reverse();
    String::from_utf8(buf).expect("base-36 is always ASCII")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string() {
        // Deterministic: same input always gives same output.
        let h = short_hash("");
        assert!(!h.is_empty());
        assert_eq!(h, short_hash(""));
    }

    #[test]
    fn different_inputs_differ() {
        assert_ne!(short_hash("hello"), short_hash("world"));
    }

    #[test]
    fn handles_unicode() {
        let h = short_hash("Hello 🙈 World");
        assert!(!h.is_empty());
    }
}
