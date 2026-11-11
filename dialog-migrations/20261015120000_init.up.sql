-- Initial schema for skw-dialog-service.
-- All ids are uuidv7: creation time is uuid_extract_timestamp(id), ORDER BY id is chronological.

CREATE TYPE message_role AS ENUM ('System', 'User', 'Assistant');

-- Assistant answers are produced asynchronously (LLM calls can exceed the gate's 30s timeout).
CREATE TYPE message_status AS ENUM ('Pending', 'Done', 'Failed');

CREATE TABLE dialogs (
    id          UUID PRIMARY KEY DEFAULT uuidv7(),
    owner_id    UUID NOT NULL, -- auth entity id (= API-KEY of signed requests); lives in another schema, so no FK
    title       TEXT,
    model       TEXT NOT NULL, -- LLM model used for answers in this dialog
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now() -- last activity (new message), for "recent dialogs" ordering
);

CREATE INDEX dialogs_owner_id_updated_at_idx ON dialogs(owner_id, updated_at DESC);

CREATE TABLE messages (
    id             UUID PRIMARY KEY DEFAULT uuidv7(),
    dialog_id      UUID NOT NULL REFERENCES dialogs(id) ON DELETE CASCADE,
    role           message_role NOT NULL,
    status         message_status NOT NULL DEFAULT 'Done',
    content        TEXT NOT NULL DEFAULT '',
    error          TEXT, -- set when status = 'Failed'
    attempts       INTEGER NOT NULL DEFAULT 0, -- LLM calls made for a Pending assistant message (retry limit)
    input_tokens   INTEGER, -- LLM usage, Assistant messages only
    output_tokens  INTEGER
);

CREATE INDEX messages_dialog_id_id_idx ON messages(dialog_id, id);
