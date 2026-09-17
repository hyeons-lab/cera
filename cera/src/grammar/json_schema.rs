//! JSON Schema to GBNF compiler for structured outputs.
//!
//! Converts JSON Schema specifications (Draft 7 and 2020-12 subsets) into GBNF
//! grammars suitable for compilation with [`super::Grammar::parse`]. Constrains
//! LLM generation to strictly valid JSON matching the schema.

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

struct SchemaCompiler {
    rule_counter: usize,
    rules: Vec<(String, String)>,
    defs: HashMap<String, Value>,
}

impl SchemaCompiler {
    fn new() -> Self {
        Self {
            rule_counter: 0,
            rules: Vec::new(),
            defs: HashMap::new(),
        }
    }

    fn extract_defs(&mut self, root: &Value) {
        if let Some(obj) = root.as_object() {
            if let Some(Value::Object(defs)) = obj.get("definitions") {
                for (k, v) in defs {
                    self.defs.insert(k.clone(), v.clone());
                }
            }
            if let Some(Value::Object(defs)) = obj.get("$defs") {
                for (k, v) in defs {
                    self.defs.insert(k.clone(), v.clone());
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
        // Handle $ref
        if let Some(r) = schema.get("$ref").and_then(|v| v.as_str()) {
            let def_name = r
                .strip_prefix("#/$defs/")
                .or_else(|| r.strip_prefix("#/definitions/"))
                .unwrap_or(r);
            if let Some(target) = self.defs.get(def_name).cloned() {
                let rule_name = format!("json-ref-{}", sanitize_identifier(def_name));
                if !self.rules.iter().any(|(n, _)| n == &rule_name) {
                    // Placeholder to avoid infinite recursion on circular refs
                    self.rules.push((rule_name.clone(), String::new()));
                    let expr = self.compile_value(&target)?;
                    if let Some(entry) = self.rules.iter_mut().find(|(n, _)| n == &rule_name) {
                        entry.1 = expr;
                    }
                }
                return Ok(rule_name);
            } else {
                bail!("unresolved $ref: {r}");
            }
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
            let mut exprs = Vec::new();
            for v in variants {
                exprs.push(self.compile_value(v)?);
            }
            return Ok(format!("( {} )", exprs.join(" | ")));
        }

        // Handle allOf
        if let Some(Value::Array(subschemas)) = schema.get("allOf") {
            ensure!(!subschemas.is_empty(), "allOf array must not be empty");
            if subschemas.len() == 1
                && schema.get("properties").is_none()
                && schema.get("required").is_none()
            {
                return self.compile_value(&subschemas[0]);
            }
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
                // If sub has $ref, resolve it
                let resolved_sub = if let Some(r) = sub.get("$ref").and_then(|v| v.as_str()) {
                    let def_name = r
                        .strip_prefix("#/$defs/")
                        .or_else(|| r.strip_prefix("#/definitions/"))
                        .unwrap_or(r);
                    if let Some(target) = self.defs.get(def_name) {
                        target.clone()
                    } else {
                        bail!("unresolved $ref in allOf: {r}");
                    }
                } else {
                    sub.clone()
                };

                if let Some(obj) = resolved_sub.as_object() {
                    if let Some(Value::Object(p)) = obj.get("properties") {
                        for (k, v) in p {
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

        let min_items = schema.get("minItems").and_then(|v| v.as_u64()).unwrap_or(0);

        let expr = if min_items > 0 {
            format!(
                "\"[\" json-ws {item_expr} ( json-ws \",\" json-ws {item_expr} )* json-ws \"]\""
            )
        } else {
            format!(
                "\"[\" json-ws ( {item_expr} ( json-ws \",\" json-ws {item_expr} )* )? json-ws \"]\""
            )
        };

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

        if props.is_empty() {
            let expr = "\"{\" json-ws ( json-string json-ws \":\" json-ws json-value ( json-ws \",\" json-ws json-string json-ws \":\" json-ws json-value )* )? json-ws \"}\"".into();
            self.rules.push((rule_name.clone(), expr));
            return Ok(rule_name);
        }

        let required_keys: Vec<String> = schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default();

        let all_required = props.keys().all(|k| required_keys.contains(k));

        if all_required && !props.is_empty() {
            // Strict ordered representation: every property in declared order
            let mut prop_seq = Vec::new();
            for (key, prop_schema) in props {
                let val_expr = self.compile_value(prop_schema)?;
                let key_lit = gbnf_quoted_json_string(key);
                prop_seq.push(format!("{key_lit} json-ws \":\" json-ws {val_expr}"));
            }
            let inner = prop_seq.join(" json-ws \",\" json-ws ");
            let expr = format!("\"{{\" json-ws {inner} json-ws \"}}\"");
            self.rules.push((rule_name.clone(), expr));
            return Ok(rule_name);
        }

        // Permissive / optional properties representation:
        // Object containing comma-separated key-value pairs matching any of the properties
        let pair_rule = self.next_rule_name("json-pair");
        let mut pair_alts = Vec::new();
        for (key, prop_schema) in props {
            let val_expr = self.compile_value(prop_schema)?;
            let key_lit = gbnf_quoted_json_string(key);
            pair_alts.push(format!("{key_lit} json-ws \":\" json-ws {val_expr}"));
        }
        self.rules.push((pair_rule.clone(), pair_alts.join(" | ")));

        let expr = format!(
            "\"{{\" json-ws ( {pair_rule} ( json-ws \",\" json-ws {pair_rule} )* )? json-ws \"}}\""
        );
        self.rules.push((rule_name.clone(), expr));
        Ok(rule_name)
    }
}

fn sanitize_identifier(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
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
}
