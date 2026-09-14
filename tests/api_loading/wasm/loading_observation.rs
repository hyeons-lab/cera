// Appended only to the isolated mirror; no observations are production exports.
impl GenerativeModel {
    pub fn core_for_loading_probe(&self) -> &cera::GenerativeModel {
        &self.inner
    }
}
