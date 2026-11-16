//! Exact, read-only transformer K/V lookup for model-bound forward reuse.
//! Exports complete per-layer/head tensor pages with exact identity binding.
//! Decoder fetches one complete layer at a time; every KV head for the layer
//! must be present with the same immutable session contract and an exact,
//! contiguous token span before any payload reaches attention.

use crate::{
    Digest, ExactKvIdentity, GlobalMindStore, RecordId, TensorDtype, TensorLayout,
    TensorPayload,
};
use serde_json::Value;

/// Export one complete layer of K/V pages.
/// 
/// # Errors
/// Returns an error if:
/// - `layer` is out of range (`layer >= layer_count`)
/// - `kv_head_count == 0`
/// - Any required head is missing
/// - Page token intervals don't exactly cover `[token_start, token_end)`
/// - Page identities are incompatible
pub fn export_complete_tensor_layer_exact(
    &self,
    pages: &[ExactKvIdentity],
    layer: u32,
    token_start: u64,
    token_end: u64,
) -> Result<Vec<TensorPayload>, String> {
    if pages.is_empty() || pages.len() > 4096 {
        return Err("invalid complete-layer page count".into());
    }
    let first = &pages[0];
    if layer >= first.layer_count || first.kv_head_count == 0 {
        return Err("complete-layer target outside session".into());
    }
    for page in pages {
        if page.workspace != first.workspace
            || page.session != first.session
            || page.model_id != first.model_id
            || page.model_revision != first.model_revision
            || page.model_digest != first.model_digest
            || page.request_digest != first.request_digest
            || page.prefix_digest != first.prefix_digest
            || page.tppe_version != first.tppe_version
            || page.tppe_configuration_digest != first.tppe_configuration_digest
            || page.address_contract_digest != first.address_contract_digest
            || page.tensor_contract_digest != first.tensor_contract_digest
            || page.dtype != first.dtype
            || page.layout != first.layout
            || page.layer_count != first.layer_count
            || page.kv_head_count != first.kv_head_count
            || page.head_dim != first.head_dim
            || page.layer != layer
        {
            return Err("complete K/V layer mixes incompatible identities".into());
        }
    }

    // Group pages by kv_head, verifying contiguous coverage
    let mut cursor: usize = 0;
    let mut heads: Vec<Vec<&ExactKvIdentity>> = Vec::with_capacity(first.kv_head_count as usize);
    for head in 0..first.kv_head_count {
        let start = cursor;
        while cursor < pages.len() && pages[cursor].kv_head == head {
            cursor += 1;
        }
        if cursor == start {
            return Err("complete K/V layer is missing a head".into());
        }
        heads.push(pages[start..cursor].to_vec());
    }
    if cursor != pages.len() {
        return Err("complete K/V layer contains extra or misordered pages".into());
    }

    // Fetch each head's contiguous tensor span and return the payloads
    let mut results: Vec<TensorPayload> = Vec::with_capacity(first.kv_head_count * first.head_dim as usize);
    for head in 0..first.kv_head_count {
        let head_pages = &heads[head as usize];
        // Fetch the contiguous span for this head
        let span_pages = &pages[cursor - first.kv_head_count..cursor]; // all pages for this head
        // Actually, we need to fetch per-head using the existing helper
        // Let's use fetch_contiguous_tensor_span_exact
        let fetched = self.fetch_contiguous_tensor_span_exact(
            &pages[start..cursor],
            token_start,
            token_end,
        )?;
        results.extend_from_slice(&fetched);
    }
    if cursor != pages.len() {
        return Err("complete K/V layer contains extra or misordered pages".into());
    }
    Ok(results)
}
