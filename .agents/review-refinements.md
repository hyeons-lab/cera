# Repo-local review refinements

### Pillar 1: Functional Correctness, Logic & Edge Cases
<!-- Loops append bullets here. -->
- **Rewind Claims Match State**: Overriding a rewind or checkpoint check on a backend with hidden recurrent state (short-conv, SSM): reporting Ok because the counters can move, while the device-side state cannot; refuse partial rewinds unless every stateful component is checkpointed, and verify by asserting the check errors for a stateful layer.

### Pillar 2: Security, Authentication & Input Sanitization
<!-- Loops append bullets here. -->

### Pillar 3: Concurrency, Asynchrony & Lifecycle Management
<!-- Loops append bullets here. -->
- **Refcount Process-Global Toggles**: Enabling a process-wide vote or mode (wakelock, QoS, power hint) per device or session: the first owner to drop cancels it for the rest, and a failed constructor leaks it; hold it in an RAII guard behind a global count (0 to 1 acquires, 1 to 0 releases), created before the first fallible step, and verify with a stub driver that logs acquire and release calls.
- **Recover Poison Consistently**: Fixing a poisoned-lock path in one function: sibling entry points that restage the same state still fail permanently, while recovering a lock over state that persists across calls (recurrent buffers, counters) continues over a torn pair; grep every lock site guarding the same resources, recover only state rewritten in full per call (log the recovery, drop any half-built pending work on entry), and reset persistent state explicitly on poison. A std `Mutex` stays poisoned after `into_inner()`: clear the flag (`clear_poison`) when recovering, or every later call repeats the reset and the warning; route all sites through one helper and unit-test that recovery reports once. For a lock guarding DSP-resident state that a panic can tear (the device lock), fail closed with an error until an explicit reset rewrites it, instead of recovering and continuing.

### Pillar 4: Error Handling, Resilience & Diagnostics
<!-- Loops append bullets here. -->
- **Errors Not Silent Prefixes**: A multi-step operation whose earlier stage returns bare results (logs plus zeros): dropping that result and continuing over the gap yields plausible output computed over missing state; propagate the failed stage as an error, and verify with a fault-injecting test that the caller sees Err.
- **Recover Poison When Overwriting**: Resetting state behind a mutex with `if let Ok(guard)`: a poisoned lock skips the zeroing while counters still reset; recover with `into_inner` when the guarded state is fully overwritten.

### Pillar 5: Interface Contracts, API Design & Compatibility
<!-- Loops append bullets here. -->
- **Idempotent Attach Setters**: A setter that also flips an unrelated opt-out flag (attach resets disable): callers that set the flag in config get it silently undone; keep each setter to its own field, and test the real construction path rather than assigning fields directly.
- **Docs Feed Generated Checksums**: Editing a doc comment on an exported FFI item: generated bindings embed the doc text and the interface checksum, so the drift job fails and native and binding checksums disagree; regenerate every binding target (including the separately generated ones) in the same change, and verify with the drift check.
- **Plumb Opt-In Flags**: Renaming a constructor parameter to `_unused` while adding a new default: the config flag becomes dead and the experimental path turns on for everyone; keep the flag live end to end and test with a stub where support is true and the default is false.

### Pillar 6: Performance, Resource Efficiency & Scalability
<!-- Loops append bullets here. -->
- **Unaligned By-Value Descriptors**: Casting a byte buffer (align 1) to a struct reference to patch DSP or wire descriptors: relies on allocator alignment and is UB or a panic otherwise; read and write the descriptor by value with unaligned accessors behind a range check, and verify with a test that patches at a deliberately odd offset.

### Pillar 7: Code Simplification, Clean Architecture & Maintainability
<!-- Loops append bullets here. -->
- **Doc Comment Theft**: Inserting an item between a doc comment and its item, or a `use` under one: the doc silently reattaches to the wrong item; after every insertion read the lines directly above each touched item.

### Pillar 8: Testing, Observability & Verification Invariants
<!-- Loops append bullets here. -->
- **Mirror Tests Are Vacuous**: A test that re-implements the production formula or patch loop inline and asserts its own copy passes when production drifts; extract the logic into a function the production path calls and test that, and prove non-vacuity by mutating production in a scratch copy.
- **Throttles Vacate No-Op Tests**: Adding a rate limiter or cache in front of an entry point: a test asserting the no-op result passes without running the checked logic when another test consumed the window; test the unthrottled core directly and unit-test the limiter with an injected clock.
- **Untested Fix Is Unfixed**: Fixing a defect in code needing a device or driver: the fix regresses silently; extract the pure decision (counter, chunk split, validation) so a host test can pin it, and mutate production once to prove the test fails.
