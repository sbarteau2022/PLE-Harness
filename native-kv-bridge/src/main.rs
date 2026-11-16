//! PLE bridge to the existing nested SurrealDB; tensor output is binary only.
use globalmind_runtime::{
    Digest, ExactKvIdentity, ExactNamedStateIdentity, GlobalMindStore, RecordId, StoreConfig,
    TensorDtype, TensorLayout,
};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::io::{Read, Write};

fn identity(v: &Value) -> Result<ExactKvIdentity, String> {
    let s = |name: &str| v[name].as_str().ok_or_else(|| format!("missing string {name}"));
    let d = |name: &str| Digest::parse(s(name)?);
    let r = |name: &str| RecordId::parse(s(name)?);
    let n = |name: &str| v[name].as_u64().ok_or_else(|| format!("missing integer {name}"));
    let u = |name: &str| u32::try_from(n(name)?).map_err(|_| format!("integer overflow {name}"));
    let shape = v["tensor_shape"].as_array().ok_or("missing tensor shape")?;
    if shape.len() != 2 { return Err("tensor shape must have two dimensions".into()); }
    let dim = |i: usize| shape[i].as_u64().and_then(|x| u32::try_from(x).ok()).ok_or("invalid tensor dimension");
    Ok(ExactKvIdentity {
        workspace: r("workspace")?, session: r("session")?, page: r("page")?,
        model_id: s("model_id")?.into(), model_revision: s("model_revision")?.into(),
        model_digest: d("model_digest")?, request_digest: d("request_digest")?, prefix_digest: d("prefix_digest")?,
        tppe_version: s("tppe_version")?.into(), tppe_configuration_digest: d("tppe_configuration_digest")?,
        address_contract_digest: d("address_contract_digest")?, tensor_contract_digest: d("tensor_contract_digest")?,
        geometry_space: r("geometry_space")?,
        dtype: match s("dtype")? {
            "f32" => TensorDtype::F32, "f16" => TensorDtype::F16, "bf16" => TensorDtype::Bf16,
            "q8_0" => TensorDtype::Q8_0, "q4_0" => TensorDtype::Q4_0,
            _ => return Err("unsupported tensor dtype".into()),
        },
        layout: match s("layout")? {
            "token_major" => TensorLayout::TokenMajor, "head_dim_major" => TensorLayout::HeadDimMajor,
            _ => return Err("unsupported tensor layout".into()),
        },
        layer_count: u("layer_count")?, kv_head_count: u("kv_head_count")?, head_dim: u("head_dim")?,
        layer: u("layer")?, kv_head: u("kv_head")?, token_start: n("token_start")?, token_end: n("token_end")?,
        tensor_shape: [dim(0)?, dim(1)?], address: r("address")?,
        phase_state_digest: d("phase_state_digest")?, revision: u("revision")?,
        key_digest: d("key_digest")?, value_digest: d("value_digest")?,
    })
}

fn named_state_identity(v: &Value) -> Result<ExactNamedStateIdentity, String> {
    let s = |name: &str| v[name].as_str().ok_or_else(|| format!("missing string {name}"));
    let d = |name: &str| Digest::parse(s(name)?);
    let r = |name: &str| RecordId::parse(s(name)?);
    let n = |name: &str| v[name].as_u64().ok_or_else(|| format!("missing integer {name}"));
    let u = |name: &str| u32::try_from(n(name)?).map_err(|_| format!("integer overflow {name}"));
    Ok(ExactNamedStateIdentity {
        workspace: r("workspace")?,
        session: r("session")?,
        page: r("page")?,
        model_id: s("model_id")?.into(),
        model_revision: s("model_revision")?.into(),
        model_digest: d("model_digest")?,
        request_digest: d("request_digest")?,
        prefix_digest: d("prefix_digest")?,
        state_contract_digest: d("state_contract_digest")?,
        tokenizer_digest: d("tokenizer_digest")?,
        adapter_digest: d("adapter_digest")?,
        geometry_space: r("geometry_space")?,
        tppe_version: s("tppe_version")?.into(),
        address_contract_digest: d("address_contract_digest")?,
        address: r("address")?,
        layer: u("layer")?,
        name: s("name")?.into(),
        encoding: s("encoding")?.into(),
        payload_digest: d("payload_digest")?,
        revision: u("revision")?,
    })
}

fn frame(request: &[u8], keys: &[u8], values: &[u8]) -> Result<Vec<u8>, String> {
    if keys.len() + values.len() > 16 * 1024 * 1024 { return Err("tensor bridge byte budget exceeded".into()); }
    let mut out = b"PKV1".to_vec();
    out.extend_from_slice(&(keys.len() as u32).to_le_bytes());
    out.extend_from_slice(&(values.len() as u32).to_le_bytes());
    out.extend_from_slice(keys);
    out.extend_from_slice(values);
    let mut hash = Sha256::new(); hash.update(request); hash.update(&out);
    out.extend_from_slice(&hash.finalize());
    Ok(out)
}

fn named_state_response_frame(request: &[u8], payload: &[u8]) -> Result<Vec<u8>, String> {
    if payload.len() > 16 * 1024 * 1024 {
        return Err("named state response byte budget exceeded".into());
    }
    let mut out = b"NSR1".to_vec();
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    let mut hash = Sha256::new();
    hash.update(request);
    hash.update(&out);
    out.extend_from_slice(&hash.finalize());
    Ok(out)
}

fn publication(input: &[u8]) -> Result<(ExactKvIdentity, Vec<RecordId>, Digest, Vec<u8>, Vec<u8>), String> {
    if input.len() < 16 || &input[..4] != b"PKP1" { return Err("invalid publication frame".into()); }
    let n = |start| u32::from_le_bytes(input[start..start + 4].try_into().unwrap()) as usize;
    let (metadata, nk, nv) = (n(4), n(8), n(12));
    if metadata > 65536 || nk + nv > 16 * 1024 * 1024 || input.len() != 16 + metadata + nk + nv {
        return Err("publication frame exceeds bounds or has wrong lengths".into());
    }
    let v: Value = serde_json::from_slice(&input[16..16 + metadata]).map_err(|e| e.to_string())?;
    let q = identity(&v)?;
    let atoms = v["trajectory_atoms"].as_array().ok_or("missing linked atom")?;
    if atoms.len() != 1 { return Err("publication needs one selected address atom".into()); }
    let atom = RecordId::parse(atoms[0].as_str().ok_or("invalid atom ID")?)?;
    let receipt = Digest::parse(v["receipt_digest"].as_str().ok_or("missing receipt digest")?)?;
    Ok((q, vec![atom], receipt, input[16 + metadata..16 + metadata + nk].to_vec(), input[16 + metadata + nk..].to_vec()))
}

fn named_state_publication(input: &[u8]) -> Result<(ExactNamedStateIdentity, Vec<RecordId>, Digest, Vec<u8>), String> {
    if input.len() < 16 || &input[..4] != b"NSP1" {
        return Err("invalid named state publication frame".into());
    }
    let n = |start| u32::from_le_bytes(input[start..start + 4].try_into().unwrap()) as usize;
    let (metadata, np, reserved) = (n(4), n(8), n(12));
    if reserved != 0 {
        return Err("invalid reserved field in named state publication frame".into());
    }
    if metadata > 65536 || np > 16 * 1024 * 1024 || input.len() != 16 + metadata + np {
        return Err("named state publication frame exceeds bounds or has wrong lengths".into());
    }
    let v: Value = serde_json::from_slice(&input[16..16 + metadata]).map_err(|e| e.to_string())?;
    let q = named_state_identity(&v)?;
    let atoms = v["trajectory_atoms"].as_array().ok_or("missing linked atom")?;
    if atoms.len() != 1 {
        return Err("publication needs one selected address atom".into());
    }
    let atom = RecordId::parse(atoms[0].as_str().ok_or("invalid atom ID")?)?;
    let receipt = Digest::parse(v["receipt_digest"].as_str().ok_or("missing receipt digest")?)?;
    Ok((
        q,
        vec![atom],
        receipt,
        input[16 + metadata..16 + metadata + np].to_vec(),
    ))
}

fn run() -> Result<(), String> {
    let mut request = Vec::new();
    let op = std::env::args().nth(1);
    let bound = match op.as_deref() {
        None => 65536,
        Some("put") => 16 + 65536 + 16 * 1024 * 1024,
        Some("ns_put") => 16 + 65536 + 16 * 1024 * 1024,
        Some("ns_get") => 65536,
        _ => return Err("unknown bridge operation".into()),
    };
    std::io::stdin().take(bound + 1).read_to_end(&mut request).map_err(|e| e.to_string())?;
    match op.as_deref() {
        Some("put") => {
            if request.len() as u64 > bound { return Err("publication frame exceeds bound".into()); }
            let (q, atoms, receipt, key, value) = publication(&request)?;
            let store = GlobalMindStore::connect(StoreConfig::from_env())?;
            store.publish_tensor_page_exact(&q, key.into_boxed_slice(), value.into_boxed_slice(), receipt, atoms)?;
            let mut out = b"PWA1".to_vec();
            out.extend_from_slice(&Sha256::digest(&request));
            std::io::stdout().write_all(&out).map_err(|e| e.to_string())
        }
        Some("ns_put") => {
            if request.len() as u64 > bound { return Err("named state publication frame exceeds bound".into()); }
            let (q, atoms, receipt, payload) = named_state_publication(&request)?;
            let store = GlobalMindStore::connect(StoreConfig::from_env())?;
            store.publish_named_state_page_exact(&q, payload.into_boxed_slice(), receipt, atoms)?;
            let mut out = b"NWA1".to_vec();
            out.extend_from_slice(&Sha256::digest(&request));
            std::io::stdout().write_all(&out).map_err(|e| e.to_string())
        }
        Some("ns_get") => {
            if request.len() > 65536 { return Err("identity request exceeds bound".into()); }
            let parsed: Value = serde_json::from_slice(&request).map_err(|e| e.to_string())?;
            let q = named_state_identity(&parsed)?;
            let store = GlobalMindStore::connect(StoreConfig::from_env())?;
            let payload = store.fetch_named_state_payload_exact(&q)?;
            let out = named_state_response_frame(&request, &payload)?;
            std::io::stdout().write_all(&out).map_err(|e| e.to_string())
        }
        None => {
            if request.len() > 65536 { return Err("identity request exceeds bound".into()); }
            let parsed: Value = serde_json::from_slice(&request).map_err(|e| e.to_string())?;
            let q = identity(&parsed)?;
            let store = GlobalMindStore::connect(StoreConfig::from_env())?;
            let payload = store.fetch_tensor_payload_exact(&q)?;
            let out = frame(&request, &payload.key, &payload.value)?;
            std::io::stdout().write_all(&out).map_err(|e| e.to_string())
        }
        _ => Err("unknown bridge operation".into()),
    }
}

fn main() {
    if let Err(error) = run() { eprintln!("{error}"); std::process::exit(1); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_identity_is_rejected() {
        assert!(identity(&serde_json::json!({})).is_err());
    }

    #[test]
    fn malformed_publication_frames_fail_before_store_access() {
        for b in [b"".as_slice(), b"PKP1", &[0; 16], b"PKP1\xff\xff\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00"] {
            assert!(publication(b).is_err());
        }
    }

    #[test]
    fn binary_frame_binds_request_and_keeps_raw_tensor_bytes() {
        let out = frame(b"request", &[0, 255, 1], &[9, 0]).unwrap();
        assert_eq!(&out[..12], &[b'P', b'K', b'V', b'1', 3, 0, 0, 0, 2, 0, 0, 0]);
        assert_eq!(&out[12..17], &[0, 255, 1, 9, 0]);
        assert_ne!(out, frame(b"changed", &[0, 255, 1], &[9, 0]).unwrap());
    }

    #[test]
    fn malformed_named_state_frames_fail_before_store_access() {
        for b in [
            b"".as_slice(),
            b"NSP1",
            &[0; 16],
            b"NSP1\xff\xff\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00",
            b"XXXX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
            b"NSP1\x05\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00extra",
        ] {
            assert!(named_state_publication(b).is_err());
        }
    }

    #[test]
    fn named_state_response_frame_binds_request_and_payload() {
        let request = b"request";
        let payload = b"named_state_bytes_123";
        let out = named_state_response_frame(request, payload).unwrap();
        assert_eq!(&out[..4], b"NSR1");
        let len = u32::from_le_bytes(out[4..8].try_into().unwrap());
        assert_eq!(len as usize, payload.len());
        assert_eq!(&out[8..8 + payload.len()], payload);
        assert_eq!(out.len(), 8 + payload.len() + 32);

        let out_diff_req = named_state_response_frame(b"changed_request", payload).unwrap();
        assert_ne!(out, out_diff_req);
        assert_eq!(&out[..8 + payload.len()], &out_diff_req[..8 + payload.len()]);

        let out_diff_payload = named_state_response_frame(request, b"different_payload").unwrap();
        assert_ne!(out, out_diff_payload);
    }
}
