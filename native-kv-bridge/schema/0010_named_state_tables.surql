-- Named recurrent / packed state persistence tables for PLE nested mind.

DEFINE TABLE IF NOT EXISTS named_state_session SCHEMAFULL TYPE NORMAL
  COMMENT 'One model/request forward-pass named recurrent or packed state session identity.';
DEFINE FIELD IF NOT EXISTS workspace ON named_state_session TYPE record<workspace>;
DEFINE FIELD IF NOT EXISTS model_id ON named_state_session TYPE string;
DEFINE FIELD IF NOT EXISTS model_revision ON named_state_session TYPE string;
DEFINE FIELD IF NOT EXISTS model_digest ON named_state_session TYPE string;
DEFINE FIELD IF NOT EXISTS request_digest ON named_state_session TYPE string;
DEFINE FIELD IF NOT EXISTS prefix_digest ON named_state_session TYPE string;
DEFINE FIELD IF NOT EXISTS state_contract_digest ON named_state_session TYPE string;
DEFINE FIELD IF NOT EXISTS tokenizer_digest ON named_state_session TYPE string;
DEFINE FIELD IF NOT EXISTS adapter_digest ON named_state_session TYPE string;
DEFINE FIELD IF NOT EXISTS geometry_space ON named_state_session TYPE record<geometry_space>;
DEFINE FIELD IF NOT EXISTS tppe_version ON named_state_session TYPE string;
DEFINE FIELD IF NOT EXISTS address_contract_digest ON named_state_session TYPE string;
DEFINE FIELD IF NOT EXISTS active ON named_state_session TYPE bool DEFAULT true;
DEFINE FIELD IF NOT EXISTS created_at ON named_state_session TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS closed_at ON named_state_session TYPE option<datetime>;
DEFINE INDEX IF NOT EXISTS named_state_session_identity ON named_state_session
  FIELDS workspace, model_digest, request_digest, state_contract_digest UNIQUE;

DEFINE TABLE IF NOT EXISTS named_state_page SCHEMAFULL TYPE NORMAL
  COMMENT 'Named state page for one layer and named state component.';
DEFINE FIELD IF NOT EXISTS workspace ON named_state_page TYPE record<workspace>;
DEFINE FIELD IF NOT EXISTS session ON named_state_page TYPE record<named_state_session>;
DEFINE FIELD IF NOT EXISTS layer ON named_state_page TYPE int ASSERT $value >= 0;
DEFINE FIELD IF NOT EXISTS name ON named_state_page TYPE string;
DEFINE FIELD IF NOT EXISTS payload ON named_state_page TYPE bytes;
DEFINE FIELD IF NOT EXISTS payload_digest ON named_state_page TYPE string;
DEFINE FIELD IF NOT EXISTS payload_bytes ON named_state_page TYPE int ASSERT $value > 0;
DEFINE FIELD IF NOT EXISTS encoding ON named_state_page TYPE string;
DEFINE FIELD IF NOT EXISTS address ON named_state_page TYPE record<atom_address>;
DEFINE FIELD IF NOT EXISTS trajectory_atoms ON named_state_page TYPE array<record<atom>> DEFAULT [];
DEFINE FIELD IF NOT EXISTS receipt_digest ON named_state_page TYPE string;
DEFINE FIELD IF NOT EXISTS revision ON named_state_page TYPE int ASSERT $value >= 1;
DEFINE FIELD IF NOT EXISTS active ON named_state_page TYPE bool DEFAULT true;
DEFINE FIELD IF NOT EXISTS created_at ON named_state_page TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS hits ON named_state_page TYPE int DEFAULT 0 ASSERT $value >= 0;

DEFINE INDEX IF NOT EXISTS named_state_page_identity ON named_state_page
  FIELDS workspace, session, layer, name UNIQUE;
DEFINE INDEX IF NOT EXISTS named_state_page_lookup ON named_state_page
  FIELDS workspace, session, layer, name, active;
DEFINE INDEX IF NOT EXISTS named_state_page_address ON named_state_page
  FIELDS workspace, address, active;
DEFINE INDEX IF NOT EXISTS named_state_page_receipt ON named_state_page
  FIELDS workspace, receipt_digest UNIQUE;
