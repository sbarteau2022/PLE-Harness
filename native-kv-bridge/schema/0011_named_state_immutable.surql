-- Exact named state session and page identity and payload are immutable after admission.
-- Runtime counters, closure, and active state remain mutable.

DEFINE FIELD OVERWRITE workspace ON named_state_session TYPE record<workspace> READONLY;
DEFINE FIELD OVERWRITE model_id ON named_state_session TYPE string READONLY;
DEFINE FIELD OVERWRITE model_revision ON named_state_session TYPE string READONLY;
DEFINE FIELD OVERWRITE model_digest ON named_state_session TYPE string READONLY;
DEFINE FIELD OVERWRITE request_digest ON named_state_session TYPE string READONLY;
DEFINE FIELD OVERWRITE prefix_digest ON named_state_session TYPE string READONLY;
DEFINE FIELD OVERWRITE state_contract_digest ON named_state_session TYPE string READONLY;
DEFINE FIELD OVERWRITE tokenizer_digest ON named_state_session TYPE string READONLY;
DEFINE FIELD OVERWRITE adapter_digest ON named_state_session TYPE string READONLY;
DEFINE FIELD OVERWRITE geometry_space ON named_state_session TYPE record<geometry_space> READONLY;
DEFINE FIELD OVERWRITE tppe_version ON named_state_session TYPE string READONLY;
DEFINE FIELD OVERWRITE address_contract_digest ON named_state_session TYPE string READONLY;
DEFINE FIELD OVERWRITE created_at ON named_state_session TYPE datetime DEFAULT time::now() READONLY;

DEFINE FIELD OVERWRITE workspace ON named_state_page TYPE record<workspace> READONLY;
DEFINE FIELD OVERWRITE session ON named_state_page TYPE record<named_state_session> READONLY;
DEFINE FIELD OVERWRITE layer ON named_state_page TYPE int ASSERT $value >= 0 READONLY;
DEFINE FIELD OVERWRITE name ON named_state_page TYPE string READONLY;
DEFINE FIELD OVERWRITE payload ON named_state_page TYPE bytes READONLY;
DEFINE FIELD OVERWRITE payload_digest ON named_state_page TYPE string READONLY;
DEFINE FIELD OVERWRITE payload_bytes ON named_state_page TYPE int ASSERT $value > 0 READONLY;
DEFINE FIELD OVERWRITE encoding ON named_state_page TYPE string READONLY;
DEFINE FIELD OVERWRITE address ON named_state_page TYPE record<atom_address> READONLY;
DEFINE FIELD OVERWRITE trajectory_atoms ON named_state_page TYPE array<record<atom>> DEFAULT [] READONLY;
DEFINE FIELD OVERWRITE receipt_digest ON named_state_page TYPE string READONLY;
DEFINE FIELD OVERWRITE revision ON named_state_page TYPE int ASSERT $value >= 1 READONLY;
DEFINE FIELD OVERWRITE created_at ON named_state_page TYPE datetime DEFAULT time::now() READONLY;
