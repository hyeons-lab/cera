//! Renders the web stub: a second Dart file with the same public API as the
//! real bindings, minus `dart:ffi`.
//!
//! # Why this exists
//!
//! The generated bindings import `dart:ffi` unconditionally, so any app that
//! targets the web and depends on them fails to **compile** — not at run time,
//! at compile time, with `Dart library 'dart:ffi' is not available on this
//! platform`. A conditional export in the package's entry point cannot help on
//! its own, because the web branch still has to supply every name the native
//! branch does. This renders that branch.
//!
//! # The property that matters
//!
//! The stub's public surface must match the native file's **exactly**. Anything
//! missing here is a name that resolves on native and not on web, which is the
//! same compile error in a smaller box; anything extra is a name that resolves
//! on web and not on native, which is worse because web is the platform nobody
//! builds first.
//!
//! Two things keep the surfaces in step. Records, enums, error types and
//! callback interfaces are rendered by the *same* functions the native file
//! uses ([`render_data_models`], [`render_callback_interfaces`]) rather than by
//! a copy, so they cannot drift. Objects and top-level functions are rendered
//! here, but iterate the same lists and apply the same skip rules; a test
//! asserts member-for-member parity against the native output.

use std::collections::HashMap;

use super::config::CustomTypeConfig;
use super::*;

/// Name of the private helper every stubbed body calls.
const UNSUPPORTED_FN: &str = "_unsupportedOnWeb";

/// Renders the body of a member that cannot work without `dart:ffi`.
///
/// Shared with [`render_data_models`] so record and enum methods throw the same
/// way object methods do.
pub(super) fn render_unsupported_method_body(owner: &str, member: &str) -> String {
    format!("    {UNSUPPORTED_FN}('{owner}.{member}');\n")
}

/// Renders the whole `<namespace>_web.dart` file.
pub(super) fn render_web_stub(ctx: &RenderContext<'_>) -> String {
    let RenderContext {
        module_name,
        namespace_docstring,
        custom_types,
        ref api_overrides,
        functions,
        objects,
        callback_interfaces,
        records,
        enums,
        ..
    } = *ctx;

    let mut body = String::new();

    // Records, enums, errors: the same renderer the native file uses. The error
    // *lift* helpers are native-only (they decode a RustBuffer), so they are off
    // here; they are private, so leaving them out does not change the surface.
    body.push_str(&render_data_models(
        records,
        enums,
        callback_interfaces,
        false,
        custom_types,
        MethodDispatch::Unsupported,
    ));
    body.push_str(&render_callback_interfaces(
        callback_interfaces,
        custom_types,
    ));

    body.push_str(&render_stub_objects(
        objects,
        api_overrides,
        records,
        enums,
        custom_types,
    ));
    body.push_str(&render_stub_functions(
        functions,
        objects,
        records,
        enums,
        api_overrides,
        custom_types,
    ));

    let mut out = String::new();
    out.push_str(&render_stub_header(module_name, namespace_docstring));
    // Imports are decided from the rendered text rather than by re-deriving
    // which types appear: the type mapping lives in one place
    // (`map_uniffi_type_to_dart`) and a second guess at it here is exactly the
    // kind of copy this file exists to avoid. Each entry pairs a library with
    // the identifiers that require it, so adding a new one is a one-line change
    // rather than a new branch.
    for (uri, markers) in [
        ("dart:typed_data", &["Uint8List"][..]),
        ("dart:async", &["Future<", "Stream<"][..]),
        ("dart:convert", &["jsonEncode", "jsonDecode", "utf8."][..]),
    ] {
        if markers.iter().any(|marker| body.contains(marker)) {
            out.push_str(&format!("import '{uri}';\n"));
        }
    }
    out.push('\n');
    out.push_str(&render_unsupported_helper(module_name));
    out.push_str(&body);
    out
}

fn render_stub_header(module_name: &str, namespace_docstring: Option<&str>) -> String {
    let mut out = String::new();
    out.push_str("// GENERATED CODE - DO NOT EDIT BY HAND\n");
    out.push_str("//\n");
    out.push_str(&format!(
        "// Web stub for `{module_name}`. Mirrors the public API of the generated\n"
    ));
    out.push_str("// bindings so that a package exporting them conditionally still COMPILES on\n");
    out.push_str("// targets without `dart:ffi`. Every entry point throws `UnsupportedError`.\n");
    out.push_str("//\n");
    out.push_str("// Data types (records, enums, errors) are real, not stubs: they are plain\n");
    out.push_str("// Dart and carry no FFI, so constructing and inspecting them works here.\n");
    out.push('\n');
    if let Some(docstring) = namespace_docstring {
        out.push_str(&render_doc_comment(Some(docstring), ""));
    }
    out.push_str("library;\n\n");
    out
}

fn render_unsupported_helper(module_name: &str) -> String {
    let mut out = String::new();
    out.push_str("/// Thrown by every API in this stub.\n");
    out.push_str("///\n");
    out.push_str("/// Reaching one of these means the code ran on a target without `dart:ffi`\n");
    out.push_str("/// (the web). The failure is deliberate and total: there is no partial\n");
    out.push_str("/// support to fall back to.\n");
    out.push_str(&format!("Never {UNSUPPORTED_FN}(String api) {{\n"));
    out.push_str("  throw UnsupportedError(\n");
    out.push_str(&format!(
        "    '$api is not available on this platform: package `{module_name}` needs '\n"
    ));
    out.push_str("    'dart:ffi, which the web does not provide.',\n");
    out.push_str("  );\n");
    out.push_str("}\n\n");
    out
}

fn render_stub_objects(
    objects: &[UdlObject],
    api_overrides: &ApiOverrides,
    records: &[UdlRecord],
    enums: &[UdlEnum],
    custom_types: &HashMap<String, CustomTypeConfig>,
) -> String {
    let mut out = String::new();
    for object in objects {
        if api_overrides.excluded(&ApiOverrides::object_key(&object.name)) {
            continue;
        }
        let object_name = api_overrides
            .renamed_or_default(&ApiOverrides::object_key(&object.name), || {
                to_upper_camel(&object.name)
            });

        out.push('\n');
        out.push_str(&render_doc_comment(object.docstring.as_deref(), ""));

        if object.has_callback_interface {
            // A `[Trait, WithForeign]` interface is an abstract class on the
            // native side too, and Dart code implements it. Nothing about that
            // needs FFI, so the stub is the same declaration: user
            // implementations keep compiling on web even though nothing can
            // call them.
            out.push_str(&format!("abstract interface class {object_name} {{\n"));
            for method in &object.methods {
                if is_uniffi_trait_method_name(&method.name) {
                    continue;
                }
                if api_overrides
                    .excluded(&ApiOverrides::object_member_key(&object.name, &method.name))
                {
                    continue;
                }
                let (method_name, signature_return, args) =
                    stub_method_signature(object, method, api_overrides, enums, custom_types);
                out.push_str(&render_doc_comment(method.docstring.as_deref(), "  "));
                out.push_str(&format!("  {signature_return} {method_name}({args});\n"));
            }
            out.push_str("}\n");
            out.push_str(&render_stub_codec(&object_name));
            continue;
        }

        out.push_str(&format!("final class {object_name} {{\n"));
        out.push_str(&format!("  {object_name}._();\n\n"));
        // Present on every native handle class, so present here.
        out.push_str(&format!(
            "  bool get isClosed => {UNSUPPORTED_FN}('{object_name}.isClosed');\n\n"
        ));
        out.push_str(&format!(
            "  void close() => {UNSUPPORTED_FN}('{object_name}.close');\n"
        ));

        for ctor in &object.constructors {
            if api_overrides.excluded(&ApiOverrides::object_member_key(&object.name, &ctor.name)) {
                continue;
            }
            // Same skip the native renderer applies. A constructor it drops does
            // not exist there, so emitting it here would put a name on web that
            // native lacks.
            if !ctor
                .args
                .iter()
                .all(|a| is_runtime_ffi_compatible_type(&a.type_, records, enums))
            {
                continue;
            }
            let static_name = safe_dart_identifier(&api_overrides.renamed_or_default(
                &ApiOverrides::object_member_key(&object.name, &ctor.name),
                || {
                    if ctor.name == "new" {
                        "create".to_string()
                    } else {
                        to_lower_camel(&ctor.name)
                    }
                },
            ));
            let args = render_callable_args_signature(&ctor.args, enums, custom_types);
            let signature_return = if ctor.is_async {
                format!("Future<{object_name}>")
            } else {
                object_name.clone()
            };
            out.push('\n');
            out.push_str(&render_doc_comment(ctor.docstring.as_deref(), "  "));
            out.push_str(&format!(
                "  static {signature_return} {static_name}({args}) => {UNSUPPORTED_FN}('{object_name}.{static_name}');\n"
            ));
        }

        for method in &object.methods {
            if is_uniffi_trait_method_name(&method.name) {
                continue;
            }
            if api_overrides.excluded(&ApiOverrides::object_member_key(&object.name, &method.name))
            {
                continue;
            }
            if !method
                .args
                .iter()
                .all(|a| is_runtime_ffi_compatible_type(&a.type_, records, enums))
            {
                continue;
            }
            let (method_name, signature_return, args) =
                stub_method_signature(object, method, api_overrides, enums, custom_types);
            out.push('\n');
            out.push_str(&render_doc_comment(method.docstring.as_deref(), "  "));
            out.push_str(&format!(
                "  {signature_return} {method_name}({args}) => {UNSUPPORTED_FN}('{object_name}.{method_name}');\n"
            ));
        }

        out.push_str("}\n");
        out.push_str(&render_stub_codec(&object_name));
    }
    out
}

/// A handle codec, stubbed.
///
/// Not dead weight: a record with an object-typed field lowers it through
/// `<Object>FfiCodec.lower` inside the record's own `toJson`, and those records
/// are rendered here by the shared renderer. Without the codec the stub does not
/// compile, which is a confusing way to learn that a config had an interface
/// field.
fn render_stub_codec(object_name: &str) -> String {
    format!(
        "\nfinal class {object_name}FfiCodec {{\n  \
         static int lower({object_name} value) => {UNSUPPORTED_FN}('{object_name}FfiCodec.lower');\n  \
         static {object_name} lift(int handle) => {UNSUPPORTED_FN}('{object_name}FfiCodec.lift');\n}}\n"
    )
}

/// The `(name, return type, argument list)` triple for a stubbed object method,
/// derived the same way the native renderer derives it.
fn stub_method_signature(
    object: &UdlObject,
    method: &UdlObjectMethod,
    api_overrides: &ApiOverrides,
    enums: &[UdlEnum],
    custom_types: &HashMap<String, CustomTypeConfig>,
) -> (String, String, String) {
    let method_name = safe_dart_identifier(&api_overrides.renamed_or_default(
        &ApiOverrides::object_member_key(&object.name, &method.name),
        || to_lower_camel(&method.name),
    ));
    let return_type = method
        .return_type
        .as_ref()
        .map(|t| map_uniffi_type_to_dart(t, custom_types))
        .unwrap_or_else(|| "void".to_string());
    let signature_return = if method.is_async {
        format!("Future<{return_type}>")
    } else {
        return_type
    };
    let args = render_callable_args_signature(&method.args, enums, custom_types);
    (method_name, signature_return, args)
}

fn render_stub_functions(
    functions: &[UdlFunction],
    objects: &[UdlObject],
    records: &[UdlRecord],
    enums: &[UdlEnum],
    api_overrides: &ApiOverrides,
    custom_types: &HashMap<String, CustomTypeConfig>,
) -> String {
    let mut out = String::new();

    // `configureDefaultBindings` / `resetDefaultBindings` exist on the native
    // side under the same condition, so mirror that condition rather than
    // emitting them unconditionally.
    let has_runtime_functions = !functions.is_empty()
        || !objects.is_empty()
        || records.iter().any(|r| !r.methods.is_empty())
        || enums.iter().any(|e| !e.methods.is_empty());
    if has_runtime_functions {
        out.push('\n');
        out.push_str("/// Mirrors the native entry point. `dynamicLibrary` is `Object?` here\n");
        out.push_str(
            "/// because its real type is `ffi.DynamicLibrary`, which does not exist on\n",
        );
        out.push_str("/// this platform. Any call site that passes one cannot compile for web\n");
        out.push_str("/// regardless, so nothing is lost by widening it.\n");
        out.push_str(&format!(
            "void configureDefaultBindings({{Object? dynamicLibrary, String? libraryPath}}) =>\n    {UNSUPPORTED_FN}('configureDefaultBindings');\n\n"
        ));
        out.push_str(
            "/// No-op: there are no bindings to reset. Deliberately does not throw, so\n",
        );
        out.push_str("/// that teardown in a `finally` cannot mask the original error.\n");
        out.push_str("void resetDefaultBindings() {}\n");
    }

    for f in functions {
        if api_overrides.excluded(&ApiOverrides::fn_key(&f.name)) {
            continue;
        }
        let public_fn_name = safe_dart_identifier(
            &api_overrides
                .renamed_or_default(&ApiOverrides::fn_key(&f.name), || to_lower_camel(&f.name)),
        );
        let value_return_type = f
            .return_type
            .as_ref()
            .map(|t| map_uniffi_type_to_dart(t, custom_types))
            .unwrap_or_else(|| "void".to_string());
        let signature_return_type = if f.is_async {
            format!("Future<{value_return_type}>")
        } else {
            value_return_type
        };
        let args = render_callable_args_signature(&f.args, enums, custom_types);
        out.push('\n');
        out.push_str(&render_doc_comment(f.docstring.as_deref(), ""));
        out.push_str(&format!(
            "{signature_return_type} {public_fn_name}({args}) => {UNSUPPORTED_FN}('{public_fn_name}');\n"
        ));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniffi_bindgen::interface::Type;

    fn arg(name: &str, type_: Type) -> UdlArg {
        UdlArg {
            name: name.to_string(),
            type_,
            docstring: None,
            default: None,
        }
    }

    fn method(
        name: &str,
        is_async: bool,
        return_type: Option<Type>,
        args: Vec<UdlArg>,
    ) -> UdlObjectMethod {
        UdlObjectMethod {
            name: name.to_string(),
            ffi_symbol: Some(format!("ffi_{name}")),
            ffi_arg_types: vec![],
            ffi_return_type: None,
            ffi_has_rust_call_status: true,
            runtime_unsupported: None,
            docstring: None,
            is_async,
            return_type,
            throws_type: None,
            args,
        }
    }

    fn ctor(name: &str, is_async: bool, args: Vec<UdlArg>) -> UdlObjectConstructor {
        UdlObjectConstructor {
            name: name.to_string(),
            ffi_symbol: Some(format!("ffi_ctor_{name}")),
            ffi_arg_types: vec![],
            ffi_return_type: None,
            ffi_has_rust_call_status: true,
            runtime_unsupported: None,
            docstring: None,
            is_async,
            args,
            throws_type: None,
        }
    }

    fn engine_object() -> UdlObject {
        UdlObject {
            name: "engine".to_string(),
            docstring: Some("An engine.".to_string()),
            is_error: false,
            has_callback_interface: false,
            ffi_free_symbol: Some("ffi_engine_free".to_string()),
            ffi_clone_symbol: Some("ffi_engine_clone".to_string()),
            constructors: vec![
                ctor("from_path", false, vec![arg("model_path", Type::String)]),
                ctor("from_id_async", true, vec![arg("id", Type::String)]),
            ],
            methods: vec![
                method("version", false, Some(Type::String), vec![]),
                method(
                    "generate",
                    true,
                    Some(Type::UInt32),
                    vec![arg("prompt", Type::String)],
                ),
                method("shutdown", false, None, vec![]),
            ],
            trait_methods: UdlObjectTraitMethods::default(),
        }
    }

    fn sink_object() -> UdlObject {
        UdlObject {
            name: "progress_sink".to_string(),
            docstring: None,
            is_error: false,
            has_callback_interface: true,
            ffi_free_symbol: Some("ffi_sink_free".to_string()),
            ffi_clone_symbol: Some("ffi_sink_clone".to_string()),
            constructors: vec![],
            methods: vec![method(
                "on_progress",
                false,
                None,
                vec![arg("done", Type::UInt64)],
            )],
            trait_methods: UdlObjectTraitMethods::default(),
        }
    }

    fn top_level_fn(name: &str, return_type: Option<Type>) -> UdlFunction {
        UdlFunction {
            name: name.to_string(),
            ffi_symbol: Some(format!("ffi_{name}")),
            ffi_arg_types: vec![],
            ffi_return_type: None,
            ffi_has_rust_call_status: true,
            runtime_unsupported: None,
            docstring: None,
            is_async: false,
            return_type,
            throws_type: None,
            args: vec![],
        }
    }

    fn sample_record() -> UdlRecord {
        UdlRecord {
            name: "config".to_string(),
            docstring: None,
            fields: vec![
                arg("context_size", Type::UInt32),
                arg("label", Type::String),
            ],
            methods: vec![],
            traits: vec![],
            trait_methods: UdlObjectTraitMethods::default(),
        }
    }

    fn sample_enum() -> UdlEnum {
        UdlEnum {
            name: "backend".to_string(),
            docstring: None,
            is_error: false,
            is_non_exhaustive: false,
            has_discr_type: false,
            variants: vec![
                UdlEnumVariant {
                    name: "cpu".to_string(),
                    docstring: None,
                    fields: vec![],
                    discr: None,
                },
                UdlEnumVariant {
                    name: "gpu".to_string(),
                    docstring: None,
                    fields: vec![],
                    discr: None,
                },
            ],
            methods: vec![],
            traits: vec![],
            trait_methods: UdlObjectTraitMethods::default(),
        }
    }

    fn render(objects: &[UdlObject], functions: &[UdlFunction]) -> (String, String) {
        let records = vec![sample_record()];
        let enums = vec![sample_enum()];
        let ctx = RenderContext {
            module_name: "demo",
            ffi_class_name: "DemoFfi",
            library_name: "uniffi_demo",
            namespace_docstring: None,
            local_module_path: "demo",
            uniffi_contract_version: None,
            ffi_uniffi_contract_version_symbol: None,
            api_checksums: &[],
            custom_types: &HashMap::new(),
            external_packages: &HashMap::new(),
            api_overrides: ApiOverrides::new(&HashMap::new(), &[]),
            functions,
            objects,
            callback_interfaces: &[],
            records: &records,
            enums: &enums,
        };
        (render_dart_scaffold(&ctx), render_web_stub(&ctx))
    }

    /// Drops comment lines so an assertion about *code* cannot be satisfied, or
    /// broken, by prose. The stub's own doc comments say the words "dart:ffi"
    /// and "ffi.DynamicLibrary" while explaining why neither is used, and a
    /// naive substring check reads those as violations.
    fn code_only(dart: &str) -> String {
        dart.lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The stub's whole reason to exist: it must not pull in `dart:ffi`, since a
    /// web build fails to compile the moment anything on the import graph does.
    #[test]
    fn stub_never_references_dart_ffi() {
        let (_native, stub) = render(
            &[engine_object(), sink_object()],
            &[top_level_fn("version", Some(Type::String))],
        );
        let code = code_only(&stub);
        // The import, specifically. A bare "dart:ffi" substring also matches the
        // UnsupportedError message, which names the missing library on purpose.
        assert!(
            !code.contains("import 'dart:ffi'"),
            "stub imported dart:ffi:\n{code}"
        );
        assert!(
            !code.contains("ffi."),
            "stub referenced an ffi. symbol:\n{code}"
        );
        assert!(
            !code.contains("_bindings()"),
            "stub called through to the native bindings:\n{code}"
        );
        assert!(
            !code.contains("Finalizer"),
            "stub kept a native handle finalizer:\n{code}"
        );
        // Guard the guard: `code_only` must not be stripping everything, or the
        // assertions above pass on an empty string.
        assert!(
            code.contains("final class Engine {"),
            "code_only stripped real code:\n{code}"
        );
    }

    /// Every object member the native file exposes has to exist here too, or the
    /// conditional export gives an app a name that resolves on one platform and
    /// not the other.
    #[test]
    fn stub_mirrors_every_object_member() {
        let (native, stub) = render(&[engine_object(), sink_object()], &[]);
        for member in [
            "fromPath",
            "fromIdAsync",
            "version",
            "generate",
            "shutdown",
            "onProgress",
            "isClosed",
            "close",
        ] {
            assert!(
                native.contains(member),
                "test is wrong: {member} missing from native"
            );
            assert!(stub.contains(member), "stub is missing {member}:\n{stub}");
        }
        assert!(stub.contains("final class Engine {"));
        assert!(stub.contains("abstract interface class ProgressSink {"));
    }

    /// Signatures, not just names: a member that takes different arguments or
    /// returns a different type is as broken as a missing one, and less obvious.
    #[test]
    fn stub_keeps_native_signatures() {
        let (native, stub) = render(&[engine_object()], &[]);
        for signature in [
            "static Engine fromPath(String modelPath)",
            "static Future<Engine> fromIdAsync(String id)",
            "String version()",
            "Future<int> generate(String prompt)",
            "void shutdown()",
        ] {
            assert!(
                native.contains(signature),
                "test is wrong: native lacks `{signature}`"
            );
            assert!(
                stub.contains(signature),
                "stub lacks `{signature}`:\n{stub}"
            );
        }
    }

    /// Records and enums come from the shared renderer, so they are real classes
    /// here rather than throwing shells. Constructing one on the web is harmless
    /// and lets calling code stay platform-agnostic up to the first engine call.
    #[test]
    fn stub_keeps_data_types_usable() {
        let (_native, stub) = render(&[], &[]);
        assert!(stub.contains("class Config {"));
        assert!(stub.contains("const Config({"));
        assert!(stub.contains("enum Backend {"));
        // A real body, not a throw.
        assert!(stub.contains("Map<String, dynamic> toJson()"));
        assert!(!stub.contains("_unsupportedOnWeb('Config"));
    }

    /// Top-level functions are stubbed, and named in the error so a stack trace
    /// says which call failed rather than just "unsupported".
    #[test]
    fn stub_functions_throw_and_name_themselves() {
        let (_native, stub) = render(
            &[],
            &[top_level_fn("cpu_backend_report", Some(Type::String))],
        );
        assert!(
            stub.contains("String cpuBackendReport() => _unsupportedOnWeb('cpuBackendReport');")
        );
    }

    /// A constructor the native renderer skips must be skipped here too:
    /// emitting it would put a name on web that native does not have, which is
    /// the failure mode that is hardest to notice.
    #[test]
    fn stub_skips_what_native_skips() {
        let mut object = engine_object();
        // A callback-interface argument has no native FFI representation
        // (`map_runtime_native_ffi_type` returns None), so the native renderer
        // drops this constructor with a warning comment instead of binding it.
        // Object-typed arguments do NOT qualify: they lower to a `ffi.Uint64`
        // handle and bind fine.
        object.constructors.push(ctor(
            "from_handle",
            false,
            vec![arg(
                "other",
                Type::CallbackInterface {
                    name: "watcher".to_string(),
                    module_path: "demo".to_string(),
                },
            )],
        ));
        let (native, stub) = render(&[object], &[]);
        assert!(
            !native.contains("static Engine fromHandle("),
            "test is wrong: native bound a constructor it was expected to skip"
        );
        assert!(
            !stub.contains("fromHandle"),
            "stub emitted a constructor the native bindings skip:\n{stub}"
        );
    }

    /// `resetDefaultBindings` is a teardown call; making it throw would let it
    /// mask the real error when used in a `finally`.
    #[test]
    fn stub_reset_bindings_is_a_no_op() {
        let (_native, stub) = render(&[engine_object()], &[]);
        assert!(stub.contains("void resetDefaultBindings() {}"));
        assert!(stub.contains("_unsupportedOnWeb('configureDefaultBindings')"));
    }
}
