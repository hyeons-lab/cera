//! P0.1 executable contract and legacy decode boundary regressions.
//! Validates the public `cera::session::chat` coordinator against the standalone contract harness.

use cera as core_api;

#[path = "api_chat/contract.rs"]
mod contract;
#[path = "api_chat/fixtures.rs"]
mod fixtures;
#[path = "api_chat/tests.rs"]
mod tests;
