//! JSON Schema to GBNF compiler for structured outputs.
//!
//! Converts JSON Schema specifications (Draft 7 and 2020-12 subsets) into GBNF
//! grammars suitable for compilation with [`super::Grammar::parse`]. Constrains
//! generation to prefixes accepted by the compiled subset. A token limit or
//! cancellation can still leave incomplete JSON. This is not a general schema
//! validator: numeric bounds, string length/pattern/format checks and other
//! unimplemented validation keywords are not enforced.
//!
//! Object keys are emitted once in the compiler's fixed map order, including any optional keys.
//! Array bounds up to 1024 are supported. `allOf` supports a single scalar wrapper
//! or compatible object property/required merges; other intersections return an error.
//! `oneOf` compiles as alternatives, without checking exclusive branch matching.
//! `enum`, `const`, `anyOf` and `oneOf` do not combine sibling type, object or
//! array constraints; place applicable constraints inside each alternative.

use std::collections::HashMap;

use anyhow::{Result, bail, ensure};
use serde_json::Value;

/// Compile a JSON Schema string into a GBNF grammar string.
pub fn json_schema_to_gbnf_str(schema_str: &str) -> Result<String> {
    let schema: Value = serde_json::from_str(schema_str)
        .map_err(|e| anyhow::anyhow!("failed to parse JSON schema: {e}"))?;
    json_schema_to_gbnf(&schema)
}

/// Compile a serde_json Value representing a JSON Schema into a GBNF grammar string.
pub fn json_schema_to_gbnf(schema: &Value) -> Result<String> {
    let mut compiler = SchemaCompiler::new();
    compiler.extract_defs(schema);
    compiler.compile(schema)
}

const MAX_REF_DEPTH: usize = 32;

struct SchemaCompiler {
    rule_counter: usize,
    rules: Vec<(String, String)>,
    defs: HashMap<String, Value>,
    ref_rules: HashMap<String, String>,
}

impl SchemaCompiler {
    fn new() -> Self {
        Self {
            rule_counter: 0,
            rules: Vec::new(),
            defs: HashMap::new(),
            ref_rules: HashMap::new(),
        }
    }

    fn extract_defs(&mut self, root: &Value) {
        if let Some(obj) = root.as_object() {
            if let Some(Value::Object(defs)) = obj.get("definitions") {
                for (k, v) in defs {
                    self.defs
                        .insert(format!("#/definitions/{}", pointer_key(k)), v.clone());
                }
            }
            if let Some(Value::Object(defs)) = obj.get("$defs") {
                for (k, v) in defs {
                    self.defs
                        .insert(format!("#/$defs/{}", pointer_key(k)), v.clone());
                }
            }
        }
    }

    fn next_rule_name(&mut self, prefix: &str) -> String {
        self.rule_counter += 1;
        format!("{prefix}-{}", self.rule_counter)
    }

    fn compile(&mut self, root_schema: &Value) -> Result<String> {
        let root_expr = self.compile_value(root_schema)?;

        let mut out = String::new();
        out.push_str(&format!("root ::= json-ws {root_expr} json-ws\n"));

        for (name, expr) in &self.rules {
            out.push_str(&format!("{name} ::= {expr}\n"));
        }

        out.push_str(COMMON_JSON_RULES);
        Ok(out)
    }

    fn compile_value(&mut self, schema: &Value) -> Result<String> {
        // Dispatch below implements alternatives, not arbitrary intersections.
        // Reject competing siblings before a $ref/enum/union can bypass allOf.
        if schema.get("allOf").is_some() {
            ensure!(
                schema.get("allOf").is_some_and(Value::is_array),
                "allOf must be an array"
            );
            ensure!(
                schema
                    .as_object()
                    .is_some_and(|obj| obj.keys().all(|key| matches!(
                        key.as_str(),
                        "allOf"
                            | "type"
                            | "properties"
                            | "required"
                            | "additionalProperties"
                            | "$defs"
                            | "definitions"
                            | "$schema"
                            | "$id"
                            | "title"
                            | "description"
                            | "default"
                            | "examples"
                    ))),
                "unsupported sibling constraint on allOf"
            );
        }
        // Handle $ref
        if let Some(r) = schema.get("$ref").and_then(|v| v.as_str()) {
            ensure!(
                schema
                    .as_object()
                    .is_some_and(|obj| obj.keys().all(|key| matches!(
                        key.as_str(),
                        "$ref"
                            | "$defs"
                            | "definitions"
                            | "$schema"
                            | "$id"
                            | "title"
                            | "description"
                            | "default"
                            | "examples"
                    ))),
                "$ref with sibling constraints is not supported"
            );
            if let Some(rule) = self.ref_rules.get(r) {
                return Ok(rule.clone());
            }
            let target = self
                .defs
                .get(r)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("unresolved $ref: {r}"))?;
            let rule = self.next_rule_name("json-ref");
            // Register before descending so recursive references share this rule.
            self.ref_rules.insert(r.to_owned(), rule.clone());
            let index = self.rules.len();
            self.rules.push((rule.clone(), String::new()));
            let expr = self.compile_value(&target)?;
            self.rules[index].1 = expr;
            return Ok(rule);
        }

        // Handle enum
        if let Some(Value::Array(variants)) = schema.get("enum") {
            ensure!(!variants.is_empty(), "enum array must not be empty");
            let mut lits = Vec::new();
            for v in variants {
                if let Some(lit) = format_json_literal(v) {
                    lits.push(lit);
                } else {
                    bail!("unsupported non-scalar enum variant: {v}");
                }
            }
            return Ok(format!("( {} )", lits.join(" | ")));
        }

        // Handle anyOf / oneOf
        if let Some(Value::Array(variants)) = schema.get("anyOf").or_else(|| schema.get("oneOf")) {
            ensure!(!variants.is_empty(), "union array must not be empty");
            // A bare alternation cannot enforce sibling constraints: fail closed
            // like the allOf/$ref paths above instead of silently dropping them.
            ensure!(
                schema.get("anyOf").is_none() || schema.get("oneOf").is_none(),
                "anyOf and oneOf cannot be combined"
            );
            ensure!(
                schema
                    .as_object()
                    .is_some_and(|obj| obj.keys().all(|key| matches!(
                        key.as_str(),
                        "anyOf"
                            | "oneOf"
                            | "$defs"
                            | "definitions"
                            | "$schema"
                            | "$id"
                            | "title"
                            | "description"
                            | "default"
                            | "examples"
                    ))),
                "anyOf/oneOf with sibling constraints is not supported; place constraints inside each alternative"
            );
            let mut exprs = Vec::new();
            for v in variants {
                exprs.push(self.compile_value(v)?);
            }
            return Ok(format!("( {} )", exprs.join(" | ")));
        }

        // Handle allOf
        if let Some(Value::Array(subschemas)) = schema.get("allOf") {
            ensure!(!subschemas.is_empty(), "allOf array must not be empty");
            // If allOf contains a single non-object scalar (or a $ref to one), compile directly
            if subschemas.len() == 1
                && schema.get("properties").is_none()
                && schema.get("required").is_none()
                && subschemas[0].get("properties").is_none()
                && subschemas[0].get("required").is_none()
            {
                let mut target = &subschemas[0];
                let mut depth = 0;
                while let Some(r) = target.get("$ref").and_then(|v| v.as_str()) {
                    depth += 1;
                    if depth > MAX_REF_DEPTH {
                        break;
                    }
                    if let Some(t) = self.defs.get(r) {
                        target = t;
                    } else {
                        break;
                    }
                }
                if target
                    .get("type")
                    .and_then(|t| t.as_str())
                    .is_some_and(|t| t != "object")
                    && target.get("properties").is_none()
                    && target.get("required").is_none()
                {
                    ensure!(
                        schema
                            .as_object()
                            .is_some_and(|obj| obj.keys().all(|key| matches!(
                                key.as_str(),
                                "allOf"
                                    | "$defs"
                                    | "definitions"
                                    | "$schema"
                                    | "$id"
                                    | "title"
                                    | "description"
                                    | "default"
                                    | "examples"
                            ))),
                        "scalar allOf with sibling constraints is not supported"
                    );
                    return self.compile_value(&subschemas[0]);
                }
            }
            ensure!(
                schema.get("type").is_none_or(|t| t == "object"),
                "allOf intersections with non-object types are not supported"
            );
            // Merge the supported object intersection subset. Reject constraints
            // that flattening would discard instead of broadening the schema.
            validate_object_intersection(schema, true)?;
            // Merge subschemas and any sibling root properties into a unified object definition
            let mut merged = serde_json::Map::new();
            let mut merged_props = serde_json::Map::new();
            let mut merged_required = Vec::new();

            if let Some(Value::Object(p)) = schema.get("properties") {
                for (k, v) in p {
                    merged_props.insert(k.clone(), v.clone());
                }
            }
            if let Some(Value::Array(r)) = schema.get("required") {
                for item in r {
                    if !merged_required.contains(item) {
                        merged_required.push(item.clone());
                    }
                }
            }

            for sub in subschemas {
                // Resolve $ref chains recursively, accumulating sibling properties and required keys
                let mut current = sub;
                let mut depth = 0;

                loop {
                    validate_object_intersection(current, false)?;
                    if let Some(obj) = current.as_object() {
                        if let Some(Value::Object(p)) = obj.get("properties") {
                            for (k, v) in p {
                                ensure!(
                                    merged_props.get(k).is_none_or(|old| old == v),
                                    "intersecting different schemas for property {k} is not supported"
                                );
                                merged_props.insert(k.clone(), v.clone());
                            }
                        }
                        if let Some(Value::Array(r)) = obj.get("required") {
                            for item in r {
                                if !merged_required.contains(item) {
                                    merged_required.push(item.clone());
                                }
                            }
                        }
                    }

                    let Some(r) = current.get("$ref").and_then(|v| v.as_str()) else {
                        break;
                    };

                    depth += 1;
                    if depth > MAX_REF_DEPTH {
                        bail!(
                            "exceeded maximum $ref depth ({MAX_REF_DEPTH}) in allOf: possible circular reference"
                        );
                    }
                    if let Some(target) = self.defs.get(r) {
                        current = target;
                    } else {
                        bail!("unresolved $ref in allOf: {r}");
                    }
                }
            }
            merged.insert("type".to_string(), Value::String("object".to_string()));
            merged.insert("properties".to_string(), Value::Object(merged_props));
            if !merged_required.is_empty() {
                merged.insert("required".to_string(), Value::Array(merged_required));
            }
            return self.compile_object(&Value::Object(merged));
        }

        // Handle const
        if let Some(const_val) = schema.get("const") {
            if let Some(lit) = format_json_literal(const_val) {
                return Ok(lit);
            } else {
                bail!("unsupported non-scalar const value: {const_val}");
            }
        }

        // Handle types
        let ty = schema.get("type").and_then(|t| t.as_str());
        match ty {
            Some("string") => Ok("json-string".into()),
            Some("integer") => Ok("json-integer".into()),
            Some("number") => Ok("json-number".into()),
            Some("boolean") => Ok("json-boolean".into()),
            Some("null") => Ok("json-null".into()),
            Some("array") => self.compile_array(schema),
            Some("object") => self.compile_object(schema),
            None => {
                // If properties or additionalProperties exist, treat as object
                if schema.get("properties").is_some() || schema.get("required").is_some() {
                    self.compile_object(schema)
                } else if schema.get("items").is_some() {
                    self.compile_array(schema)
                } else {
                    // Fallback to any valid JSON value
                    Ok("json-value".into())
                }
            }
            Some(other) => bail!("unsupported JSON schema type: {other}"),
        }
    }

    fn compile_array(&mut self, schema: &Value) -> Result<String> {
        let rule_name = self.next_rule_name("json-arr");
        let item_expr = if let Some(items) = schema.get("items") {
            self.compile_value(items)?
        } else {
            "json-value".into()
        };

        let bound = |name: &str| -> Result<Option<usize>> {
            schema
                .get(name)
                .map(|value| {
                    let value = value
                        .as_u64()
                        .ok_or_else(|| anyhow::anyhow!("{name} must be a nonnegative integer"))?;
                    ensure!(value <= 1024, "{name} above 1024 is not supported");
                    Ok(value as usize)
                })
                .transpose()
        };
        let min = bound("minItems")?.unwrap_or(0);
        let max = bound("maxItems")?;
        ensure!(
            max.is_none_or(|max| min <= max),
            "minItems exceeds maxItems"
        );
        let separator_item = format!(" json-ws \",\" json-ws {item_expr}");
        let inner = if max == Some(0) {
            String::new()
        } else {
            let mut inner = item_expr.clone();
            for _ in 1..min {
                inner.push_str(&separator_item);
            }
            match max {
                Some(max) => {
                    let extra = max - min.max(1);
                    if extra > 0 {
                        inner.push_str(&format!(" ( {separator_item} ){{0,{extra}}}"));
                    }
                }
                None => inner.push_str(&format!(" ( {separator_item} )*")),
            }
            if min == 0 {
                format!("( {inner} )?")
            } else {
                inner
            }
        };
        let expr = format!("\"[\" json-ws {inner} json-ws \"]\"");

        self.rules.push((rule_name.clone(), expr));
        Ok(rule_name)
    }

    fn compile_object(&mut self, schema: &Value) -> Result<String> {
        let rule_name = self.next_rule_name("json-obj");
        let empty_props = serde_json::Map::new();
        let props = schema
            .get("properties")
            .and_then(|p| p.as_object())
            .unwrap_or(&empty_props);

        let required = schema
            .get("required")
            .map(|value| {
                value
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("required must be an array"))?
                    .iter()
                    .map(|key| {
                        key.as_str()
                            .ok_or_else(|| anyhow::anyhow!("required keys must be strings"))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        ensure!(
            required.iter().all(|key| props.contains_key(*key)),
            "required keys without a properties schema are not supported"
        );
        if props.is_empty() {
            let expr = if schema.get("additionalProperties") == Some(&Value::Bool(false)) {
                "\"{\" json-ws \"}\"".into()
            } else {
                "json-object".into()
            };
            self.rules.push((rule_name.clone(), expr));
            return Ok(rule_name);
        }

        // Emit keys once in declaration order. Two suffix states distinguish an
        // empty prefix from a prefix that needs a comma, preserving required keys
        // across any combination of omitted optional properties in linear space.
        let mut first = "\"\"".to_owned();
        let mut rest = "\"\"".to_owned();
        for (key, value) in props.iter().rev() {
            let val = self.compile_value(value)?;
            let pair = format!(
                "{} json-ws \":\" json-ws {val}",
                gbnf_quoted_json_string(key)
            );
            let first_name = self.next_rule_name("json-field");
            let rest_name = self.next_rule_name("json-field");
            let first_present = format!("{pair} {rest}");
            let rest_present = format!("json-ws \",\" json-ws {pair} {rest}");
            let (first_expr, rest_expr) = if required.contains(&key.as_str()) {
                (first_present, rest_present)
            } else {
                (
                    format!("{first_present} | {first}"),
                    format!("{rest_present} | {rest}"),
                )
            };
            self.rules.push((first_name.clone(), first_expr));
            self.rules.push((rest_name.clone(), rest_expr));
            first = first_name;
            rest = rest_name;
        }
        self.rules.push((
            rule_name.clone(),
            format!("\"{{\" json-ws {first} json-ws \"}}\""),
        ));
        Ok(rule_name)
    }
}

fn pointer_key(key: &str) -> String {
    key.replace('~', "~0").replace('/', "~1")
}

fn validate_object_intersection(schema: &Value, root: bool) -> Result<()> {
    let object = schema
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("allOf requires object schemas"))?;
    ensure!(
        schema.get("type").is_none_or(|ty| ty == "object"),
        "allOf intersections with non-object types are not supported"
    );
    for (key, value) in object {
        ensure!(
            matches!(
                key.as_str(),
                "type"
                    | "properties"
                    | "required"
                    | "$ref"
                    | "$defs"
                    | "definitions"
                    | "$schema"
                    | "$id"
                    | "title"
                    | "description"
                    | "default"
                    | "examples"
            ) || (root && key == "allOf")
                || (key == "additionalProperties" && value == &Value::Bool(true)),
            "unsupported constraint in object allOf: {key}"
        );
    }
    Ok(())
}

fn gbnf_lit(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

fn gbnf_quoted_json_string(s: &str) -> String {
    let json = serde_json::to_string(s).unwrap_or_else(|_| format!("\"{s}\""));
    gbnf_lit(&json)
}

fn format_json_literal(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(gbnf_quoted_json_string(s)),
        Value::Bool(true) => Some("\"true\"".into()),
        Value::Bool(false) => Some("\"false\"".into()),
        Value::Null => Some("\"null\"".into()),
        Value::Number(n) => Some(gbnf_lit(&n.to_string())),
        _ => None,
    }
}

const COMMON_JSON_RULES: &str = r#"json-value ::= json-object | json-array | json-string | json-number | json-boolean | json-null
json-object ::= "{" json-ws ( json-string json-ws ":" json-ws json-value ( json-ws "," json-ws json-string json-ws ":" json-ws json-value )* )? json-ws "}"
json-array ::= "[" json-ws ( json-value ( json-ws "," json-ws json-value )* )? json-ws "]"
json-string ::= "\"" json-char* "\""
json-char ::= [^"\\\x00-\x1F] | "\\" json-esc
json-esc ::= ["\\/bfnrt] | "u" [0-9a-fA-F]{4}
json-integer ::= "-"? ("0" | [1-9] [0-9]*)
json-number ::= json-integer ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
json-boolean ::= "true" | "false"
json-null ::= "null"
json-ws ::= [ \t\n\r]*
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::{Grammar, GrammarState};
    use serde_json::json;

    #[test]
    fn primitive_types_compile_and_parse() {
        for ty in ["string", "integer", "number", "boolean", "null"] {
            let schema = json!({ "type": ty });
            let gbnf = json_schema_to_gbnf(&schema).expect("compiles cleanly");
            Grammar::parse(&gbnf).expect("gbnf parses as valid grammar");
        }
    }

    #[test]
    fn enum_strings_compile_and_parse() {
        let schema = json!({
            "type": "string",
            "enum": ["asc", "desc", "auto"]
        });
        let gbnf = json_schema_to_gbnf(&schema).expect("compiles cleanly");
        assert!(gbnf.contains(r#""\"asc\"" | "\"desc\"" | "\"auto\""#));
        Grammar::parse(&gbnf).expect("valid grammar");
    }

    #[test]
    fn strict_object_with_required_properties() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" },
                "is_active": { "type": "boolean" }
            },
            "required": ["name", "age", "is_active"],
            "additionalProperties": false
        });
        let gbnf = json_schema_to_gbnf(&schema).expect("compiles cleanly");
        assert!(gbnf.contains(r#""\"name\"""#));
        assert!(gbnf.contains(r#""\"age\"""#));
        assert!(gbnf.contains(r#""\"is_active\"""#));
        Grammar::parse(&gbnf).expect("valid grammar");
    }

    #[test]
    fn nested_arrays_and_objects() {
        let schema = json!({
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": { "type": "integer" },
                            "tags": {
                                "type": "array",
                                "items": { "type": "string" }
                            }
                        },
                        "required": ["id"]
                    }
                }
            }
        });
        let gbnf = json_schema_to_gbnf(&schema).expect("compiles cleanly");
        Grammar::parse(&gbnf).expect("valid grammar");
    }

    #[test]
    fn definitions_and_ref_resolution() {
        let schema = json!({
            "$defs": {
                "Coordinates": {
                    "type": "object",
                    "properties": {
                        "lat": { "type": "number" },
                        "lon": { "type": "number" }
                    },
                    "required": ["lat", "lon"]
                }
            },
            "type": "object",
            "properties": {
                "location": { "$ref": "#/$defs/Coordinates" }
            },
            "required": ["location"]
        });
        let gbnf = json_schema_to_gbnf(&schema).expect("compiles cleanly");
        Grammar::parse(&gbnf).expect("valid grammar");
    }

    #[test]
    fn union_any_of_compilation() {
        let schema = json!({
            "anyOf": [
                { "type": "string" },
                { "type": "integer" },
                { "type": "null" }
            ]
        });
        let gbnf = json_schema_to_gbnf(&schema).expect("compiles cleanly");
        Grammar::parse(&gbnf).expect("valid grammar");
    }

    #[test]
    fn union_with_sibling_constraints_fails() {
        // Sibling constraints on a bare alternation would be silently dropped.
        for schema in [
            json!({"anyOf": [{"type": "string"}], "properties": {}}),
            json!({"oneOf": [{"type": "string"}], "type": "string"}),
            json!({"anyOf": [{"type": "string"}], "oneOf": [{"type": "string"}]}),
        ] {
            assert!(json_schema_to_gbnf(&schema).is_err());
        }
        // Pure unions with annotation-only siblings still compile.
        let schema = json!({"title": "u", "anyOf": [{"type": "string"}]});
        json_schema_to_gbnf(&schema).expect("compiles cleanly");
    }

    #[test]
    fn invalid_ref_fails() {
        let schema = json!({
            "$ref": "#/$defs/NonExistent"
        });
        assert!(json_schema_to_gbnf(&schema).is_err());
    }

    #[test]
    fn empty_enum_fails() {
        let schema = json!({
            "enum": []
        });
        assert!(json_schema_to_gbnf(&schema).is_err());
    }

    #[test]
    fn test_grammar_state_accepts_quote() {
        let gbnf = r#"root ::= json-ws ( "\"a\"" ) json-ws
json-ws ::= [ \t\n\r]*
"#;
        let g = Grammar::parse(gbnf).unwrap();
        let mut state = GrammarState::new(std::sync::Arc::new(g));
        assert!(state.accepts(b"\""));
        assert!(state.accepts(b" "));
        state.accept(b" ");
        assert!(state.accepts(b"\""));
    }

    #[test]
    fn all_of_single_ref_wrapper() {
        let schema = json!({
            "$defs": {
                "Coordinates": {
                    "type": "object",
                    "properties": {
                        "lat": { "type": "number" },
                        "lon": { "type": "number" }
                    },
                    "required": ["lat", "lon"]
                }
            },
            "allOf": [
                { "$ref": "#/$defs/Coordinates" }
            ]
        });
        let gbnf = json_schema_to_gbnf(&schema).expect("compiles cleanly");
        Grammar::parse(&gbnf).expect("valid grammar");
    }

    #[test]
    fn all_of_merged_schemas() {
        let schema = json!({
            "$defs": {
                "Base": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" }
                    },
                    "required": ["id"]
                }
            },
            "allOf": [
                { "$ref": "#/$defs/Base" },
                {
                    "properties": {
                        "name": { "type": "string" }
                    },
                    "required": ["name"]
                }
            ]
        });
        let gbnf = json_schema_to_gbnf(&schema).expect("compiles cleanly");
        Grammar::parse(&gbnf).expect("valid grammar");
        assert!(gbnf.contains(r#"\"id\""#));
        assert!(gbnf.contains(r#"\"name\""#));
    }

    #[test]
    fn empty_all_of_fails() {
        let schema = json!({
            "allOf": []
        });
        assert!(json_schema_to_gbnf(&schema).is_err());
    }

    #[test]
    fn all_of_unresolved_ref_fails() {
        let schema = json!({
            "allOf": [
                { "$ref": "#/$defs/DoesNotExist" }
            ]
        });
        assert!(json_schema_to_gbnf(&schema).is_err());
    }

    #[test]
    fn all_of_sibling_properties_merging() {
        let schema = json!({
            "$defs": {
                "Base": {
                    "type": "object",
                    "properties": {
                        "base_field": { "type": "string" }
                    },
                    "required": ["base_field"]
                }
            },
            "allOf": [
                { "$ref": "#/$defs/Base" }
            ],
            "properties": {
                "sibling_field": { "type": "integer" }
            },
            "required": ["sibling_field"]
        });
        let gbnf = json_schema_to_gbnf(&schema).expect("compiles cleanly");
        Grammar::parse(&gbnf).expect("valid grammar");
        assert!(gbnf.contains(r#"\"base_field\""#));
        assert!(gbnf.contains(r#"\"sibling_field\""#));
    }

    #[test]
    fn all_of_chained_refs() {
        let schema = json!({
            "$defs": {
                "Leaf": {
                    "type": "object",
                    "properties": {
                        "leaf_val": { "type": "boolean" }
                    },
                    "required": ["leaf_val"]
                },
                "Intermediate": {
                    "$ref": "#/$defs/Leaf"
                }
            },
            "allOf": [
                { "$ref": "#/$defs/Intermediate" }
            ]
        });
        let gbnf = json_schema_to_gbnf(&schema).expect("compiles cleanly");
        Grammar::parse(&gbnf).expect("valid grammar");
        assert!(gbnf.contains(r#"\"leaf_val\""#));
    }

    #[test]
    fn all_of_sibling_properties_on_ref_element() {
        let schema = json!({
            "$defs": {
                "Base": {
                    "type": "object",
                    "properties": {
                        "base_val": { "type": "string" }
                    },
                    "required": ["base_val"]
                }
            },
            "allOf": [
                {
                    "$ref": "#/$defs/Base",
                    "properties": {
                        "extra_val": { "type": "number" }
                    },
                    "required": ["extra_val"]
                }
            ]
        });
        let gbnf = json_schema_to_gbnf(&schema).expect("compiles cleanly");
        Grammar::parse(&gbnf).expect("valid grammar");
        assert!(gbnf.contains(r#"\"base_val\""#));
        assert!(gbnf.contains(r#"\"extra_val\""#));
    }

    #[test]
    fn all_of_circular_ref_fails() {
        let schema = json!({
            "$defs": {
                "LoopA": {
                    "$ref": "#/$defs/LoopB"
                },
                "LoopB": {
                    "$ref": "#/$defs/LoopA"
                }
            },
            "allOf": [
                { "$ref": "#/$defs/LoopA" }
            ]
        });
        assert!(json_schema_to_gbnf(&schema).is_err());
    }

    #[test]
    fn all_of_single_ref_to_scalar() {
        let schema = json!({
            "$defs": {
                "MyString": {
                    "type": "string"
                }
            },
            "allOf": [
                { "$ref": "#/$defs/MyString" }
            ]
        });
        let gbnf = json_schema_to_gbnf(&schema).expect("compiles cleanly");
        Grammar::parse(&gbnf).expect("valid grammar");
        assert!(gbnf.contains("json-string"));
    }
}
