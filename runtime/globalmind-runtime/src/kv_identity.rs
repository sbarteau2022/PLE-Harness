//! Exact immutable transformer K/V publication and lookup for model-bound reuse.

use crate::store::{record, record_ref, sq};
use crate::{Digest, GlobalMindStore, RecordId, TensorDtype, TensorLayout, TensorPayload, TensorSession, TensorPage};
use serde_json::Value;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactKvIdentity {
    pub workspace: RecordId,
    pub session: RecordId,
    pub page: RecordId,
    pub model_id: String,
    pub model_revision: String,
    pub model_digest: Digest,
    pub request_digest: Digest,
    pub prefix_digest: Digest,
    pub tppe_version: String,
    pub tppe_configuration_digest: Digest,
    pub address_contract_digest: Digest,
    pub tensor_contract_digest: Digest,
    pub geometry_space: RecordId,
    pub dtype: TensorDtype,
    pub layer_count: u32,
    pub kv_head_count: u32,
    pub head_dim: u32,
    pub layer: u32,
    pub kv_head: u32,
    pub token_start: u64,
    pub token_end: u64,
    pub tensor_shape: [u32; 2],
    pub layout: TensorLayout,
    pub address: RecordId,
    pub phase_state_digest: Digest,
    pub revision: u32,
    pub key_digest: Digest,
    pub value_digest: Digest,
}

fn required_str<'a>(row: &'a Value, field: &str) -> Result<&'a str, String> {
    row.get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("tensor roster missing {field}"))
}

fn required_u32(row: &Value, field: &str) -> Result<u32, String> {
    row.get(field)
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| format!("tensor roster invalid {field}"))
}

fn required_u64(row: &Value, field: &str) -> Result<u64, String> {
    row.get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("tensor roster invalid {field}"))
}

fn required_record(row: &Value, field: &str, table: &str) -> Result<RecordId, String> {
    let raw = required_str(row, field)?;
    let id = raw
        .strip_prefix(&format!("{table}:"))
        .ok_or_else(|| format!("tensor roster {field} has wrong record table"))?;
    RecordId::parse(id)
}

fn parse_dtype(row: &Value) -> Result<TensorDtype, String> {
    match required_str(row, "dtype")? {
        "f16" => Ok(TensorDtype::F16),
        "bf16" => Ok(TensorDtype::Bf16),
        "f32" => Ok(TensorDtype::F32),
        "q8_0" => Ok(TensorDtype::Q8_0),
        "q4_0" => Ok(TensorDtype::Q4_0),
        _ => Err("tensor roster dtype is unknown".into()),
    }
}

fn parse_layout(row: &Value) -> Result<TensorLayout, String> {
    match required_str(row, "layout")? {
        "token_major" => Ok(TensorLayout::TokenMajor),
        "head_dim_major" => Ok(TensorLayout::HeadDimMajor),
        _ => Err("tensor roster layout is unknown".into()),
    }
}

impl ExactKvIdentity {
    fn validate(&self) -> Result<(), String> {
        if self.model_id.trim().is_empty()
            || self.model_revision.trim().is_empty()
            || self.tppe_version.trim().is_empty()
            || self.layer_count == 0
            || self.kv_head_count == 0
            || self.head_dim == 0
            || self.layer >= self.layer_count
            || self.kv_head >= self.kv_head_count
            || self.token_end <= self.token_start
            || self.revision == 0
            || self.tensor_shape[0] as u64 != self.token_end - self.token_start
            || self.tensor_shape[1] != self.head_dim
        {
            return Err("invalid exact tensor K/V identity".into());
        }
        Ok(())
    }
}

fn exactly_one_id(rows: &Value, expected: &str) -> Result<(), String> {
    let rows = rows
        .as_array()
        .ok_or_else(|| format!("{expected} lookup returned no array"))?;
    if rows.len() != 1 {
        return Err(format!("cache miss or ambiguous {expected} identity"));
    }
    if rows[0]["id"].as_str() != Some(expected) {
        return Err(format!("{expected} identity changed during lookup"));
    }
    Ok(())
}

impl GlobalMindStore {
    /// Reconstruct the immutable exact K/V roster from SurrealDB after a
    /// process restart. This reads metadata only; the caller must compare
    /// the returned contract with its currently pinned model and fetch each
    /// payload through `fetch_complete_tensor_layer_exact`.
    pub fn lookup_exact_tensor_roster(
        &self,
        workspace: &RecordId,
        model_digest: &Digest,
        prefix_digest: &Digest,
    ) -> Result<Option<Vec<ExactKvIdentity>>, String> {
        let sessions = self.one(&format!(
            "SELECT * FROM transformer_kv_session WHERE active = true AND workspace = {workspace} AND model_digest = {model} AND prefix_digest = {prefix} LIMIT 2;",
            workspace = record("workspace", workspace),
            model = sq(model_digest.as_str()),
            prefix = sq(prefix_digest.as_str()),
        ))?;
        let sessions = sessions
            .as_array()
            .ok_or("tensor session lookup is not an array")?;
        if sessions.is_empty() {
            return Ok(None);
        }
        if sessions.len() != 1 {
            return Err("ambiguous exact tensor session identity".into());
        }
        let s = &sessions[0];
        if required_str(s, "workspace")? != record_ref("workspace", workspace)
            || required_str(s, "model_digest")? != model_digest.as_str()
            || required_str(s, "prefix_digest")? != prefix_digest.as_str()
        {
            return Err("tensor session lookup changed identity".into());
        }
        let session = required_record(s, "id", "transformer_kv_session")?;
        let layer_count = required_u32(s, "layer_count")?;
        let kv_head_count = required_u32(s, "kv_head_count")?;
        let head_dim = required_u32(s, "head_dim")?;
        let minimum = usize::try_from(layer_count)
            .ok()
            .and_then(|l| {
                usize::try_from(kv_head_count)
                    .ok()
                    .and_then(|h| l.checked_mul(h))
            })
            .ok_or("tensor roster dimensions overflow")?;
        if minimum == 0 || minimum > 4096 || head_dim == 0 {
            return Err("tensor roster dimensions out of bounds".into());
        }
        let dtype = parse_dtype(s)?;
        let model_id = required_str(s, "model_id")?.to_string();
        let model_revision = required_str(s, "model_revision")?.to_string();
        let request_digest = Digest::parse(required_str(s, "request_digest")?)?;
        let tppe_version = required_str(s, "toroidal_phase_encoder_version")?.to_string();
        let tppe_configuration_digest =
            Digest::parse(required_str(s, "tppe_configuration_digest")?)?;
        let address_contract_digest = Digest::parse(required_str(s, "address_contract_digest")?)?;
        let tensor_contract_digest = Digest::parse(required_str(s, "tensor_contract_digest")?)?;
        let geometry_space = required_record(s, "geometry_space", "geometry_space")?;
        let rows = self.one(&format!(
            "SELECT id, workspace, session, layer, kv_head, token_start, token_end, tensor_shape, dtype, layout, address, phase_state_digest, revision, key_digest, value_digest FROM transformer_kv_page WHERE active = true AND workspace = {workspace} AND session = {session} ORDER BY layer, kv_head, token_start LIMIT 4097;",
            workspace = record("workspace", workspace),
            session = record("transformer_kv_session", &session),
        ))?;
        let rows = rows
            .as_array()
            .ok_or("tensor page lookup is not an array")?;
        if rows.len() < minimum || rows.len() > 4096 {
            return Err("incomplete or oversized tensor page roster".into());
        }
        let mut roster = Vec::with_capacity(rows.len());
        let mut seen = vec![false; minimum];
        for row in rows {
            if required_str(row, "workspace")? != record_ref("workspace", workspace)
                || required_str(row, "session")? != record_ref("transformer_kv_session", &session)
                || parse_dtype(row)? != dtype
            {
                return Err("tensor page roster contains foreign identity".into());
            }
            let layer = required_u32(row, "layer")?;
            let kv_head = required_u32(row, "kv_head")?;
            if layer >= layer_count || kv_head >= kv_head_count {
                return Err("tensor page roster contains out-of-range head".into());
            }
            seen[layer as usize * kv_head_count as usize + kv_head as usize] = true;
            let shape = row
                .get("tensor_shape")
                .and_then(Value::as_array)
                .ok_or("tensor page shape missing")?;
            if shape.len() != 2 {
                return Err("tensor page shape is not rank two".into());
            }
            let tensor_shape = [
                shape[0]
                    .as_u64()
                    .and_then(|v| u32::try_from(v).ok())
                    .ok_or("tensor page token shape invalid")?,
                shape[1]
                    .as_u64()
                    .and_then(|v| u32::try_from(v).ok())
                    .ok_or("tensor page head shape invalid")?,
            ];
            let q = ExactKvIdentity {
                workspace: workspace.clone(),
                session: session.clone(),
                page: required_record(row, "id", "transformer_kv_page")?,
                model_id: model_id.clone(),
                model_revision: model_revision.clone(),
                model_digest: model_digest.clone(),
                request_digest: request_digest.clone(),
                prefix_digest: prefix_digest.clone(),
                tppe_version: tppe_version.clone(),
                tppe_configuration_digest: tppe_configuration_digest.clone(),
                address_contract_digest: address_contract_digest.clone(),
                tensor_contract_digest: tensor_contract_digest.clone(),
                geometry_space: geometry_space.clone(),
                dtype,
                layer_count,
                kv_head_count,
                head_dim,
                layer,
                kv_head,
                token_start: required_u64(row, "token_start")?,
                token_end: required_u64(row, "token_end")?,
                tensor_shape,
                layout: parse_layout(row)?,
                address: required_record(row, "address", "atom_address")?,
                phase_state_digest: Digest::parse(required_str(row, "phase_state_digest")?)?,
                revision: required_u32(row, "revision")?,
                key_digest: Digest::parse(required_str(row, "key_digest")?)?,
                value_digest: Digest::parse(required_str(row, "value_digest")?)?,
            };
            q.validate()?;
            roster.push(q);
        }
        if seen.iter().any(|present| !present) {
            return Err("tensor page roster is missing a layer or KV head".into());
        }
        Ok(Some(roster))
    }

    /// Return K/V only when every model, request, TPPE, coordinate and payload
    /// digest field matches the caller's immutable witness.
    fn require_tensor_session_exact(&self, q: &ExactKvIdentity) -> Result<(), String> {
        q.validate()?;
        let session = record("transformer_kv_session", &q.session);
        let session_rows = self.one(&format!(
            "SELECT id FROM {session} WHERE active = true AND workspace = {workspace} \
             AND model_id = {model_id} AND model_revision = {model_revision} \
             AND model_digest = {model_digest} AND request_digest = {request_digest} \
             AND prefix_digest = {prefix_digest} AND toroidal_phase_encoder_version = {tppe_version} \
             AND tppe_configuration_digest = {tppe_config} AND address_contract_digest = {address_contract} \
             AND tensor_contract_digest = {tensor_contract} AND geometry_space = {geometry_space} \
             AND dtype = {dtype} AND layer_count = {layer_count} AND kv_head_count = {kv_head_count} \
             AND head_dim = {head_dim} LIMIT 2;",
            workspace = record("workspace", &q.workspace),
            model_id = sq(&q.model_id),
            model_revision = sq(&q.model_revision),
            model_digest = sq(q.model_digest.as_str()),
            request_digest = sq(q.request_digest.as_str()),
            prefix_digest = sq(q.prefix_digest.as_str()),
            tppe_version = sq(&q.tppe_version),
            tppe_config = sq(q.tppe_configuration_digest.as_str()),
            address_contract = sq(q.address_contract_digest.as_str()),
            tensor_contract = sq(q.tensor_contract_digest.as_str()),
            geometry_space = record("geometry_space", &q.geometry_space),
            dtype = sq(q.dtype.as_str()),
            layer_count = q.layer_count,
            kv_head_count = q.kv_head_count,
            head_dim = q.head_dim,
        ))?;
        exactly_one_id(
            &session_rows,
            &format!("transformer_kv_session:{}", q.session),
        )?;
        Ok(())
    }

    /// Append one immutable native page, resuming an identical partial upload.
    /// Existing conflicting sessions/pages are never updated or replaced.
    pub fn publish_tensor_page_exact(&self, q: &ExactKvIdentity, key: Box<[u8]>, value: Box<[u8]>,
                                     receipt: Digest, atoms: Vec<RecordId>) -> Result<(), String> {
        q.validate()?;
        if key.len().saturating_add(value.len()) > 16 * 1024 * 1024 {
            return Err("tensor publication byte budget exceeded".into());
        }
        for payload in [&key, &value] {
            let finite = match q.dtype {
                TensorDtype::F32 => payload.chunks_exact(4).all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
                TensorDtype::F16 => payload.chunks_exact(2).all(|b| u16::from_le_bytes(b.try_into().unwrap()) & 0x7c00 != 0x7c00),
                TensorDtype::Bf16 => payload.chunks_exact(2).all(|b| u16::from_le_bytes(b.try_into().unwrap()) & 0x7f80 != 0x7f80),
                _ => return Err("publication needs an explicit dense scalar format".into()),
            };
            if !finite { return Err("tensor publication contains non-finite values".into()); }
        }
        let page = TensorPage {
            id: q.page.clone(), workspace: q.workspace.clone(), session: q.session.clone(),
            layer: q.layer, kv_head: q.kv_head, token_start: q.token_start, token_end: q.token_end,
            tensor_shape: q.tensor_shape, dtype: q.dtype, layout: q.layout,
            key_tensor: key, value_tensor: value, key_digest: q.key_digest.clone(), value_digest: q.value_digest.clone(),
            address: q.address.clone(), trajectory_atoms: atoms, well: None, trajectory: None,
            phase_state_digest: q.phase_state_digest.clone(), receipt_digest: receipt, revision: q.revision,
        };
        page.validate()?; // Validate bytes and shape before creating any records.
        let geometry = self.one(&format!("SELECT id FROM {} LIMIT 1;", record("geometry_space", &q.geometry_space)))?;
        exactly_one_id(&geometry, &record_ref("geometry_space", &q.geometry_space))?;
        let addresses = self.one(&format!("SELECT id, atom FROM {} WHERE active = true AND workspace = {} LIMIT 1;",
            record("atom_address", &q.address), record("workspace", &q.workspace)))?;
        exactly_one_id(&addresses, &record_ref("atom_address", &q.address))?;
        if page.trajectory_atoms.len() != 1 || addresses[0]["atom"].as_str()
                != Some(&record_ref("atom", &page.trajectory_atoms[0])) {
            return Err("tensor publication must link its selected address atom".into());
        }
        let atom = self.one(&format!("SELECT id FROM {} WHERE active = true AND workspace = {} LIMIT 1;",
            record("atom", &page.trajectory_atoms[0]), record("workspace", &q.workspace)))?;
        exactly_one_id(&atom, &record_ref("atom", &page.trajectory_atoms[0]))?;
        let session = TensorSession {
            id: q.session.clone(), workspace: q.workspace.clone(), model_id: q.model_id.clone(),
            model_revision: q.model_revision.clone(), model_digest: q.model_digest.clone(),
            request_digest: q.request_digest.clone(), prefix_digest: q.prefix_digest.clone(), dtype: q.dtype,
            layer_count: q.layer_count, kv_head_count: q.kv_head_count, head_dim: q.head_dim,
            geometry_space: q.geometry_space.clone(), tppe_version: q.tppe_version.clone(),
            tppe_configuration_digest: q.tppe_configuration_digest.clone(),
            address_contract_digest: q.address_contract_digest.clone(), tensor_contract_digest: q.tensor_contract_digest.clone(),
        };
        session.validate()?;
        let present = self.one(&format!("SELECT id FROM {} LIMIT 1;", record("transformer_kv_session", &q.session)))?;
        if present.as_array().ok_or("invalid session lookup")?.is_empty() {
            if let Err(error) = self.create_tensor_session(&session) {
                self.require_tensor_session_exact(q).map_err(|_| error)?; // Concurrent identical publisher.
            }
        }
        self.require_tensor_session_exact(q)?;
        let present = self.one(&format!("SELECT id FROM {} LIMIT 1;", record("transformer_kv_page", &q.page)))?;
        if present.as_array().ok_or("invalid page lookup")?.is_empty() {
            if let Err(error) = self.put_tensor_page(&page) {
                self.fetch_tensor_payload_exact(q).map_err(|_| error)?;
            }
        }
        self.fetch_tensor_payload_exact(q)?;
        let metadata = self.one(&format!("SELECT id, receipt_digest, trajectory_atoms FROM {} LIMIT 1;",
            record("transformer_kv_page", &q.page)))?;
        exactly_one_id(&metadata, &record_ref("transformer_kv_page", &q.page))?;
        let expected_atoms = Value::Array(page.trajectory_atoms.iter()
            .map(|id| Value::String(record_ref("atom", id))).collect());
        if metadata[0]["receipt_digest"].as_str() != Some(page.receipt_digest.as_str())
                || metadata[0]["trajectory_atoms"] != expected_atoms {
            return Err("immutable tensor publication lineage or receipt conflict".into());
        }
        Ok(())
    }

    pub fn fetch_tensor_payload_exact(&self, q: &ExactKvIdentity) -> Result<TensorPayload, String> {
        self.require_tensor_session_exact(q)?;
        let session = record("transformer_kv_session", &q.session);
        let page = record("transformer_kv_page", &q.page);

        let page_rows = self.one(&format!(
            "SELECT id FROM {page} WHERE active = true AND workspace = {workspace} AND session = {session} \
             AND layer = {layer} AND kv_head = {kv_head} AND token_start = {token_start} \
             AND token_end = {token_end} AND tensor_shape = [{shape0}, {shape1}] \
             AND dtype = {dtype} AND layout = {layout} AND address = {address} \
             AND phase_state_digest = {phase_state} AND revision = {revision} \
             AND key_digest = {key_digest} AND value_digest = {value_digest} LIMIT 2;",
            workspace = record("workspace", &q.workspace),
            layer = q.layer,
            kv_head = q.kv_head,
            token_start = q.token_start,
            token_end = q.token_end,
            shape0 = q.tensor_shape[0],
            shape1 = q.tensor_shape[1],
            dtype = sq(q.dtype.as_str()),
            layout = sq(q.layout.as_str()),
            address = record("atom_address", &q.address),
            phase_state = sq(q.phase_state_digest.as_str()),
            revision = q.revision,
            key_digest = sq(q.key_digest.as_str()),
            value_digest = sq(q.value_digest.as_str()),
        ))?;
        exactly_one_id(&page_rows, &format!("transformer_kv_page:{}", q.page))?;
        let payload = self.fetch_tensor_payload(&q.page)?;
        if Digest::sha256(&payload.key) != q.key_digest
            || Digest::sha256(&payload.value) != q.value_digest
        {
            return Err("tensor K/V payload changed after identity lookup".into());
        }
        Ok(payload)
    }

    /// Reassemble one complete per-layer/head K/V interval from exact pages.
    /// Input pages must be in token order. A missing, overlapping, or foreign
    /// page is a cache miss; no partial tensor is returned to the decoder.
    pub fn fetch_contiguous_tensor_span_exact(
        &self,
        pages: &[ExactKvIdentity],
        token_start: u64,
        token_end: u64,
    ) -> Result<TensorPayload, String> {
        const MAX_SPAN_BYTES: usize = 256 * 1024 * 1024;
        if pages.is_empty() || pages.len() > 1024 || token_end <= token_start {
            return Err("invalid exact K/V span page count or interval".into());
        }
        let first = &pages[0];
        let scalar_bytes = first
            .dtype
            .scalar_bytes()
            .ok_or("quantized K/V span assembly requires an explicit block layout")?;
        let dim = usize::try_from(first.head_dim).map_err(|_| "head dimension overflow")?;
        let total_tokens =
            usize::try_from(token_end - token_start).map_err(|_| "K/V token span overflow")?;
        let total_bytes = total_tokens
            .checked_mul(dim)
            .and_then(|n| n.checked_mul(scalar_bytes))
            .ok_or("K/V span byte count overflow")?;
        if total_bytes > MAX_SPAN_BYTES {
            return Err("K/V span exceeds 256 MiB per tensor".into());
        }

        let mut cursor = token_start;
        let mut payloads = Vec::with_capacity(pages.len());
        for page in pages {
            if page.token_start != cursor || page.token_end > token_end {
                return Err("K/V span has a gap, overlap, or out-of-range page".into());
            }
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
                || page.geometry_space != first.geometry_space
                || page.dtype != first.dtype
                || page.layer_count != first.layer_count
                || page.kv_head_count != first.kv_head_count
                || page.head_dim != first.head_dim
                || page.layer != first.layer
                || page.kv_head != first.kv_head
                || page.layout != first.layout
            {
                return Err("K/V span mixes incompatible page identities".into());
            }
            let payload = self.fetch_tensor_payload_exact(page)?;
            let page_tokens = usize::try_from(page.token_end - page.token_start)
                .map_err(|_| "K/V page token count overflow")?;
            let page_bytes = page_tokens
                .checked_mul(dim)
                .and_then(|n| n.checked_mul(scalar_bytes))
                .ok_or("K/V page byte count overflow")?;
            if payload.key.len() != page_bytes || payload.value.len() != page_bytes {
                return Err("K/V page byte count does not match its shape".into());
            }
            payloads.push(payload);
            cursor = page.token_end;
        }
        if cursor != token_end {
            return Err("K/V span does not cover the requested token interval".into());
        }

        let mut key = Vec::with_capacity(total_bytes);
        let mut value = Vec::with_capacity(total_bytes);
        match first.layout {
            TensorLayout::TokenMajor => {
                for payload in &payloads {
                    key.extend_from_slice(&payload.key);
                    value.extend_from_slice(&payload.value);
                }
            }
            TensorLayout::HeadDimMajor => {
                for coordinate in 0..dim {
                    for payload in &payloads {
                        let page_tokens = payload.key.len() / (dim * scalar_bytes);
                        let start = coordinate * page_tokens * scalar_bytes;
                        let end = start + page_tokens * scalar_bytes;
                        key.extend_from_slice(&payload.key[start..end]);
                        value.extend_from_slice(&payload.value[start..end]);
                    }
                }
            }
        }
        Ok(TensorPayload {
            key: key.into_boxed_slice(),
            value: value.into_boxed_slice(),
        })
    }

    /// Hydrate one complete attention layer, with every KV head present.
    /// Pages are ordered by head then token interval; missing heads fail closed.
    pub fn fetch_complete_tensor_layer_exact(
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
                || page.geometry_space != first.geometry_space
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
        let mut cursor = 0;
        let mut heads = Vec::with_capacity(first.kv_head_count as usize);
        for head in 0..first.kv_head_count {
            let start = cursor;
            while cursor < pages.len() && pages[cursor].kv_head == head {
                cursor += 1;
            }
            if cursor == start {
                return Err("complete K/V layer is missing a head".into());
            }
            heads.push(self.fetch_contiguous_tensor_span_exact(
                &pages[start..cursor],
                token_start,
                token_end,
            )?);
        }
        if cursor != pages.len() {
            return Err("complete K/V layer contains extra or misordered pages".into());
        }
        Ok(heads)
    }
}


#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactNamedStateIdentity {
    pub workspace: RecordId,
    pub session: RecordId,
    pub page: RecordId,
    pub model_id: String,
    pub model_revision: String,
    pub model_digest: Digest,
    pub request_digest: Digest,
    pub prefix_digest: Digest,
    pub state_contract_digest: Digest,
    pub tokenizer_digest: Digest,
    pub adapter_digest: Digest,
    pub geometry_space: RecordId,
    pub tppe_version: String,
    pub address_contract_digest: Digest,
    pub address: RecordId,
    pub layer: u32,
    pub name: String,
    pub encoding: String,
    pub payload_digest: Digest,
    pub revision: u32,
}

impl ExactNamedStateIdentity {
    pub fn validate(&self) -> Result<(), String> {
        if self.model_id.trim().is_empty()
            || self.model_revision.trim().is_empty()
            || self.tppe_version.trim().is_empty()
            || self.name.trim().is_empty()
            || self.encoding.trim().is_empty()
            || self.revision == 0
        {
            return Err("invalid exact named state identity".into());
        }
        Ok(())
    }
}

impl GlobalMindStore {
    /// Return named state only when all model, request, TPPE, coordinate and
    /// contract digest fields match the caller's immutable witness.
    fn require_named_state_session_exact(&self, q: &ExactNamedStateIdentity) -> Result<(), String> {
        q.validate()?;
        let session = record("named_state_session", &q.session);
        let session_rows = self.one(&format!(
            "SELECT id FROM {session} WHERE active = true AND workspace = {workspace} \
             AND model_id = {model_id} AND model_revision = {model_revision} \
             AND model_digest = {model_digest} AND request_digest = {request_digest} \
             AND prefix_digest = {prefix_digest} AND state_contract_digest = {state_contract} \
             AND tokenizer_digest = {tokenizer_digest} AND adapter_digest = {adapter_digest} \
             AND geometry_space = {geometry_space} AND tppe_version = {tppe_version} \
             AND address_contract_digest = {address_contract} LIMIT 2;",
            session = session,
            workspace = record("workspace", &q.workspace),
            model_id = sq(&q.model_id),
            model_revision = sq(&q.model_revision),
            model_digest = sq(q.model_digest.as_str()),
            request_digest = sq(q.request_digest.as_str()),
            prefix_digest = sq(q.prefix_digest.as_str()),
            state_contract = sq(q.state_contract_digest.as_str()),
            tokenizer_digest = sq(q.tokenizer_digest.as_str()),
            adapter_digest = sq(q.adapter_digest.as_str()),
            geometry_space = record("geometry_space", &q.geometry_space),
            tppe_version = sq(&q.tppe_version),
            address_contract = sq(q.address_contract_digest.as_str()),
        ))?;
        exactly_one_id(
            &session_rows,
            &format!("named_state_session:{}", q.session),
        )?;
        Ok(())
    }

    /// Read raw payload bytes from an active named state page.
    pub fn fetch_named_state_payload(&self, page: &RecordId) -> Result<Box<[u8]>, String> {
        let rows = self.one(&format!(
            "SELECT payload, payload_digest FROM {} WHERE active = true LIMIT 1;",
            record("named_state_page", page)
        ))?;
        let row = rows
            .as_array()
            .and_then(|rows| rows.first())
            .ok_or_else(|| "active named state page does not exist".to_string())?;
        let payload = crate::store::json_bytes(row.get("payload"), "payload")?;
        let payload_digest = row
            .get("payload_digest")
            .and_then(Value::as_str)
            .ok_or_else(|| "named state page payload digest is missing".to_string())?;
        if Digest::sha256(&payload).as_str() != payload_digest {
            return Err("stored named state page digest mismatch".into());
        }
        Ok(payload.into_boxed_slice())
    }

    /// Return named state payload only when every model, request, coordinate and
    /// payload digest field matches the caller's immutable witness.
    pub fn fetch_named_state_payload_exact(
        &self,
        q: &ExactNamedStateIdentity,
    ) -> Result<Box<[u8]>, String> {
        self.require_named_state_session_exact(q)?;
        let session = record("named_state_session", &q.session);
        let page = record("named_state_page", &q.page);

        let page_rows = self.one(&format!(
            "SELECT id FROM {page} WHERE active = true AND workspace = {workspace} AND session = {session} \
             AND layer = {layer} AND name = {name} AND encoding = {encoding} AND address = {address} \
             AND payload_digest = {payload_digest} AND revision = {revision} LIMIT 2;",
            page = page,
            workspace = record("workspace", &q.workspace),
            session = session,
            layer = q.layer,
            name = sq(&q.name),
            encoding = sq(&q.encoding),
            address = record("atom_address", &q.address),
            payload_digest = sq(q.payload_digest.as_str()),
            revision = q.revision,
        ))?;
        exactly_one_id(&page_rows, &format!("named_state_page:{}", q.page))?;
        let payload = self.fetch_named_state_payload(&q.page)?;
        if Digest::sha256(&payload) != q.payload_digest {
            return Err("named state payload changed after identity lookup".into());
        }
        Ok(payload)
    }

    /// Append one immutable named state page, resuming an identical partial upload.
    /// Existing conflicting sessions/pages are never updated or replaced.
    pub fn publish_named_state_page_exact(
        &self,
        q: &ExactNamedStateIdentity,
        payload: Box<[u8]>,
        receipt: Digest,
        atoms: Vec<RecordId>,
    ) -> Result<(), String> {
        q.validate()?;
        if payload.is_empty() || payload.len() > 16 * 1024 * 1024 {
            return Err("named state payload must be non-empty and at most 16 MiB".into());
        }
        if Digest::sha256(&payload) != q.payload_digest {
            return Err("named state payload digest mismatch".into());
        }
        let geometry = self.one(&format!(
            "SELECT id FROM {} LIMIT 1;",
            record("geometry_space", &q.geometry_space)
        ))?;
        exactly_one_id(&geometry, &record_ref("geometry_space", &q.geometry_space))?;
        let addresses = self.one(&format!(
            "SELECT id, atom FROM {} WHERE active = true AND workspace = {} LIMIT 1;",
            record("atom_address", &q.address),
            record("workspace", &q.workspace)
        ))?;
        exactly_one_id(&addresses, &record_ref("atom_address", &q.address))?;
        if atoms.len() != 1
            || addresses[0]["atom"].as_str() != Some(&record_ref("atom", &atoms[0]))
        {
            return Err("named state publication must link its selected address atom".into());
        }
        let atom = self.one(&format!(
            "SELECT id FROM {} WHERE active = true AND workspace = {} LIMIT 1;",
            record("atom", &atoms[0]),
            record("workspace", &q.workspace)
        ))?;
        exactly_one_id(&atom, &record_ref("atom", &atoms[0]))?;

        let session_id = record("named_state_session", &q.session);
        let session_present = self.one(&format!("SELECT id FROM {session_id} LIMIT 1;"))?;
        if session_present
            .as_array()
            .ok_or("invalid session lookup")?
            .is_empty()
        {
            let create_session_query = format!(
                "BEGIN TRANSACTION; \
                 CREATE {session_id} CONTENT {{ \
                     workspace: {workspace}, model_id: {model_id}, model_revision: {model_revision}, \
                     model_digest: {model_digest}, request_digest: {request_digest}, prefix_digest: {prefix_digest}, \
                     state_contract_digest: {state_contract}, tokenizer_digest: {tokenizer_digest}, \
                     adapter_digest: {adapter_digest}, geometry_space: {geometry_space}, \
                     tppe_version: {tppe_version}, address_contract_digest: {address_contract}, \
                     active: true, created_at: time::now(), closed_at: NONE \
                 }} RETURN NONE; \
                 COMMIT TRANSACTION;",
                session_id = session_id,
                workspace = record("workspace", &q.workspace),
                model_id = sq(&q.model_id),
                model_revision = sq(&q.model_revision),
                model_digest = sq(q.model_digest.as_str()),
                request_digest = sq(q.request_digest.as_str()),
                prefix_digest = sq(q.prefix_digest.as_str()),
                state_contract = sq(q.state_contract_digest.as_str()),
                tokenizer_digest = sq(q.tokenizer_digest.as_str()),
                adapter_digest = sq(q.adapter_digest.as_str()),
                geometry_space = record("geometry_space", &q.geometry_space),
                tppe_version = sq(&q.tppe_version),
                address_contract = sq(q.address_contract_digest.as_str()),
            );
            if let Err(error) = self.transaction(&create_session_query) {
                self.require_named_state_session_exact(q).map_err(|_| error)?;
            }
        }
        self.require_named_state_session_exact(q)?;

        let page_id = record("named_state_page", &q.page);
        let page_present = self.one(&format!("SELECT id FROM {page_id} LIMIT 1;"))?;
        if page_present
            .as_array()
            .ok_or("invalid page lookup")?
            .is_empty()
        {
            let atoms_str = atoms
                .iter()
                .map(|id| record("atom", id))
                .collect::<Vec<_>>()
                .join(", ");
            let create_page_query = format!(
                "BEGIN TRANSACTION; \
                 CREATE {page_id} CONTENT {{ \
                     workspace: {workspace}, session: {session}, layer: {layer}, name: {name}, \
                     payload: {payload}, payload_digest: {payload_digest}, payload_bytes: {payload_bytes}, \
                     encoding: {encoding}, address: {address}, trajectory_atoms: [{atoms}], \
                     receipt_digest: {receipt}, revision: {revision}, active: true, \
                     created_at: time::now(), hits: 0 \
                 }} RETURN NONE; \
                 COMMIT TRANSACTION;",
                page_id = page_id,
                workspace = record("workspace", &q.workspace),
                session = session_id,
                layer = q.layer,
                name = sq(&q.name),
                payload = crate::store::bytes_literal(&payload),
                payload_digest = sq(q.payload_digest.as_str()),
                payload_bytes = payload.len(),
                encoding = sq(&q.encoding),
                address = record("atom_address", &q.address),
                atoms = atoms_str,
                receipt = sq(receipt.as_str()),
                revision = q.revision,
            );
            if let Err(error) = self.transaction(&create_page_query) {
                self.fetch_named_state_payload_exact(q).map_err(|_| error)?;
            }
        }
        self.fetch_named_state_payload_exact(q)?;

        let metadata = self.one(&format!(
            "SELECT id, receipt_digest, trajectory_atoms FROM {} LIMIT 1;",
            record("named_state_page", &q.page)
        ))?;
        exactly_one_id(&metadata, &record_ref("named_state_page", &q.page))?;
        let expected_atoms = Value::Array(
            atoms
                .iter()
                .map(|id| Value::String(record_ref("atom", id)))
                .collect(),
        );
        if metadata[0]["receipt_digest"].as_str() != Some(receipt.as_str())
            || metadata[0]["trajectory_atoms"] != expected_atoms
        {
            return Err("immutable named state publication lineage or receipt conflict".into());
        }
        Ok(())
    }

    /// Reconstruct the immutable exact named state roster from SurrealDB.
    /// Returns one page per (layer, name) ordered by layer, name.
    pub fn lookup_exact_named_state_roster(
        &self,
        workspace: &RecordId,
        model_digest: &Digest,
        prefix_digest: &Digest,
        state_contract_digest: &Digest,
    ) -> Result<Option<Vec<ExactNamedStateIdentity>>, String> {
        let sessions = self.one(&format!(
            "SELECT * FROM named_state_session WHERE active = true AND workspace = {workspace} AND model_digest = {model} AND prefix_digest = {prefix} AND state_contract_digest = {state_contract} LIMIT 2;",
            workspace = record("workspace", workspace),
            model = sq(model_digest.as_str()),
            prefix = sq(prefix_digest.as_str()),
            state_contract = sq(state_contract_digest.as_str()),
        ))?;
        let sessions = sessions
            .as_array()
            .ok_or("named state session lookup is not an array")?;
        if sessions.is_empty() {
            return Ok(None);
        }
        if sessions.len() != 1 {
            return Err("ambiguous exact named state session identity".into());
        }
        let s = &sessions[0];
        if required_str(s, "workspace")? != record_ref("workspace", workspace)
            || required_str(s, "model_digest")? != model_digest.as_str()
            || required_str(s, "prefix_digest")? != prefix_digest.as_str()
            || required_str(s, "state_contract_digest")? != state_contract_digest.as_str()
        {
            return Err("named state session lookup changed identity".into());
        }
        let session = required_record(s, "id", "named_state_session")?;
        let model_id = required_str(s, "model_id")?.to_string();
        let model_revision = required_str(s, "model_revision")?.to_string();
        let request_digest = Digest::parse(required_str(s, "request_digest")?)?;
        let tokenizer_digest = Digest::parse(required_str(s, "tokenizer_digest")?)?;
        let adapter_digest = Digest::parse(required_str(s, "adapter_digest")?)?;
        let geometry_space = required_record(s, "geometry_space", "geometry_space")?;
        let tppe_version = required_str(s, "tppe_version")?.to_string();
        let address_contract_digest = Digest::parse(required_str(s, "address_contract_digest")?)?;

        let rows = self.one(&format!(
            "SELECT id, workspace, session, layer, name, encoding, address, payload_digest, revision FROM named_state_page WHERE active = true AND workspace = {workspace} AND session = {session} ORDER BY layer, name LIMIT 4097;",
            workspace = record("workspace", workspace),
            session = record("named_state_session", &session),
        ))?;
        let rows = rows
            .as_array()
            .ok_or("named state page lookup is not an array")?;
        if rows.is_empty() {
            return Ok(None);
        }
        if rows.len() > 4096 {
            return Err("oversized named state page roster".into());
        }
        let mut roster = Vec::with_capacity(rows.len());
        for row in rows {
            if required_str(row, "workspace")? != record_ref("workspace", workspace)
                || required_str(row, "session")? != record_ref("named_state_session", &session)
            {
                return Err("named state page roster contains foreign identity".into());
            }
            let layer = required_u32(row, "layer")?;
            let name = required_str(row, "name")?.to_string();
            let encoding = required_str(row, "encoding")?.to_string();
            let address = required_record(row, "address", "atom_address")?;
            let payload_digest = Digest::parse(required_str(row, "payload_digest")?)?;
            let revision = required_u32(row, "revision")?;
            let q = ExactNamedStateIdentity {
                workspace: workspace.clone(),
                session: session.clone(),
                page: required_record(row, "id", "named_state_page")?,
                model_id: model_id.clone(),
                model_revision: model_revision.clone(),
                model_digest: model_digest.clone(),
                request_digest: request_digest.clone(),
                prefix_digest: prefix_digest.clone(),
                state_contract_digest: state_contract_digest.clone(),
                tokenizer_digest: tokenizer_digest.clone(),
                adapter_digest: adapter_digest.clone(),
                geometry_space: geometry_space.clone(),
                tppe_version: tppe_version.clone(),
                address_contract_digest: address_contract_digest.clone(),
                address,
                layer,
                name,
                encoding,
                payload_digest,
                revision,
            };
            q.validate()?;
            roster.push(q);
        }
        Ok(Some(roster))
    }
}
