//! Focused behavioral probes for the 0.6 API audit.

use cera::grammar::{Grammar, GrammarState};
use serde_json::{Value, json};
use std::sync::Arc;

fn accepts_complete(schema: Value, output: &str) -> bool {
    let grammar = Grammar::from_json_schema(&schema).expect("schema compiles");
    let mut state = GrammarState::new(Arc::new(grammar));
    state.accept(output.as_bytes());
    !state.is_dead() && state.is_complete()
}

#[test]
fn required_property_remains_required_with_optional_sibling() {
    let schema = json!({
        "type": "object",
        "properties": {"id": {"type": "integer"}, "note": {"type": "string"}},
        "required": ["id"],
        "additionalProperties": false
    });
    assert!(
        !accepts_complete(schema.clone(), "{}"),
        "grammar accepts missing required id"
    );
    assert!(accepts_complete(schema.clone(), r#"{"id":1}"#));
    assert!(accepts_complete(schema.clone(), r#"{"id":1,"note":"ok"}"#));
    assert!(!accepts_complete(schema.clone(), r#"{"note":"ok"}"#));
    assert!(!accepts_complete(schema, r#"{"id":1,"id":2}"#));
}

#[test]
fn array_bounds_are_enforced() {
    let schema =
        json!({"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3});
    assert!(
        !accepts_complete(schema.clone(), "[1]"),
        "grammar accepts fewer than minItems"
    );
    assert!(accepts_complete(schema.clone(), "[1,2,3]"));
    assert!(!accepts_complete(schema.clone(), "[1,2,3,4]"));
    assert!(!accepts_complete(schema, "[]"));
}

#[test]
fn unsupported_scalar_intersection_is_rejected() {
    let schema = json!({"allOf": [{"type": "integer"}, {"minimum": 0}]});
    assert!(
        Grammar::from_json_schema(&schema).is_err(),
        "unsupported scalar allOf must fail instead of changing type"
    );
    for schema in [
        json!({"enum":[1], "allOf":[{"type":"string"}]}),
        json!({"anyOf":[{"type":"integer"}], "allOf":[{"type":"string"}]}),
        json!({"$defs":{"S":{"type":"string"}}, "allOf":[{"$ref":"#/$defs/S", "type":"integer"}]}),
        json!({"type":"string", "allOf":[{"type":"integer"}]}),
        json!({"type":"object", "allOf":[{"type":"integer"}]}),
        json!({"type":"array", "minItems":3,
            "allOf":[{"type":"array", "items":{"type":"integer"}}]}),
    ] {
        assert!(Grammar::from_json_schema(&schema).is_err());
    }
    assert!(accepts_complete(
        json!({"title":"wrapper", "allOf":[{"type":"integer"}]}),
        "1"
    ));
}

#[test]
fn distinct_definition_names_remain_distinct() {
    let schema = json!({
        "$defs": {"a-b": {"type": "integer"}, "a_b": {"type": "string"}},
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/a-b"}, "y": {"$ref": "#/$defs/a_b"}},
        "required": ["x", "y"]
    });
    assert!(
        !accepts_complete(schema.clone(), r#"{"x":1,"y":2}"#),
        "colliding definition names allow integer for string"
    );
    assert!(accepts_complete(schema, r#"{"x":1,"y":"ok"}"#));
}

#[test]
fn bounded_array_ranges_include_only_permitted_cardinalities() {
    for min in 0..=3 {
        for max in min..=4 {
            let schema =
                json!({"type":"array", "items":{"type":"integer"}, "minItems":min, "maxItems":max});
            for count in 0..=5 {
                let output = serde_json::to_string(&vec![1; count]).unwrap();
                assert_eq!(
                    accepts_complete(schema.clone(), &output),
                    (min..=max).contains(&count),
                    "min={min}, max={max}, output={output}"
                );
            }
        }
    }
    for bounds in [
        json!({"minItems":-1}),
        json!({"minItems":3,"maxItems":2}),
        json!({"maxItems":1025}),
    ] {
        let mut schema = json!({"type":"array"});
        schema
            .as_object_mut()
            .unwrap()
            .extend(bounds.as_object().unwrap().clone());
        assert!(Grammar::from_json_schema(&schema).is_err());
    }
}

#[test]
fn optional_keys_on_both_sides_preserve_required_keys_and_commas() {
    let schema = json!({"type":"object", "properties":{"a":{"type":"integer"},"b":{"type":"integer"},"c":{"type":"integer"}},"required":["b"]});
    for output in [
        r#"{"b":1}"#,
        r#"{"a":0,"b":1}"#,
        r#"{"b":1,"c":2}"#,
        r#"{"a":0,"b":1,"c":2}"#,
    ] {
        assert!(accepts_complete(schema.clone(), output), "{output}");
    }
    for output in ["{}", r#"{"a":0,"c":2}"#, r#"{,"b":1}"#, r#"{"b":1,}"#] {
        assert!(!accepts_complete(schema.clone(), output), "{output}");
    }
}

#[test]
fn references_keep_namespaces_and_pointer_escaping() {
    let schema = json!({"definitions":{"a/b":{"type":"string"}},"$defs":{"a/b":{"type":"integer"}},"type":"object","properties":{"x":{"$ref":"#/$defs/a~1b"},"y":{"$ref":"#/definitions/a~1b"}},"required":["x","y"]});
    assert!(accepts_complete(schema.clone(), r#"{"x":1,"y":"ok"}"#));
    assert!(!accepts_complete(schema, r#"{"x":1,"y":2}"#));
}
