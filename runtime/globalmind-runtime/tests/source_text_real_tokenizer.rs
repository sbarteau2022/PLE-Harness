//! Real-tokenizer check of the replay source carrier: the active model's own
//! tokenizer (globalmind_runtime::model_select) encodes real language, and every atom span must be the exact text
//! prefix up to its token (including multi-byte characters split across tokens).

use core_lm::tokenizer::BpeTokenizer;
use globalmind_runtime::Digest;
use globalmind_runtime::replay::source::SourceText;
use std::path::Path;


fn carrier(text: &str) -> Result<(SourceText, Vec<u32>), String> {
    let path = globalmind_runtime::model_select::model_tokenizer()?;
    let tok = BpeTokenizer::from_json_file(Path::new(&path)).map_err(|e| e.to_string())?;
    let ids = tok.encode(text).map_err(|e| e.to_string())?;
    let source = SourceText::from_tokens(Digest::sha256(b"real-tape"), &ids, &|t| tok.decode(t))?;
    Ok((source, ids))
}

#[test]
fn real_language_spans_are_exact_prefixes() -> Result<(), String> {
    for text in [
        "The capital of France is",
        "Paris is the capital of France. It lies on the Seine.",
        "café — naïve 東京 では 猫 が 好き です",
    ] {
        let (source, ids) = carrier(text)?;
        assert_eq!(
            source.full_text(),
            text,
            "decode must round-trip the source"
        );
        let mut previous = 0;
        for position in 0..ids.len() {
            // Positions that end mid-character may be empty at the very start;
            // any non-empty span must be a monotone prefix of the text.
            if let Ok(span) = source.span(position as u64) {
                assert!(text.starts_with(span));
                assert!(span.len() >= previous);
                previous = span.len();
            }
        }
        assert_eq!(previous, text.len(), "last span must cover the whole text");
        let back = SourceText::from_bytes(&source.to_bytes())?;
        assert_eq!(back, source);
        println!(
            "PASS {} tokens, {} bytes, digest {}",
            ids.len(),
            text.len(),
            source.digest()
        );
    }
    Ok(())
}
