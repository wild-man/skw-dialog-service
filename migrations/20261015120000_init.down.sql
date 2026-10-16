-- Revert of initial schema.

DROP TABLE IF EXISTS messages;
DROP TABLE IF EXISTS dialogs;
DROP TYPE IF EXISTS message_status;
DROP TYPE IF EXISTS message_role;
