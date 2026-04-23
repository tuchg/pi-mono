use pi_ai_rs::types::{Tool, ToolCall};
use pi_ai_rs::utils::validation::{validate_tool_arguments, validate_tool_call, ValidationError};
use serde_json::json;

fn calc_tool() -> Tool {
    Tool {
        name: "calculate".to_string(),
        description: "Evaluate a math expression".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "expression": { "type": "string" }
            },
            "required": ["expression"]
        }),
    }
}

#[test]
fn valid_arguments_pass() {
    let tool = calc_tool();
    let args = json!({"expression": "1+1"});
    let result = validate_tool_arguments(&tool, &args);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), args);
}

#[test]
fn missing_required_field_fails() {
    let tool = calc_tool();
    let args = json!({});
    let result = validate_tool_arguments(&tool, &args);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ValidationError::InvalidArguments { .. }));
}

#[test]
fn wrong_type_fails() {
    let tool = calc_tool();
    let args = json!({"expression": 42});
    let result = validate_tool_arguments(&tool, &args);
    assert!(result.is_err());
}

#[test]
fn empty_schema_accepts_anything() {
    let tool = Tool {
        name: "any".to_string(),
        description: "accepts anything".to_string(),
        parameters: json!({}),
    };
    let args = json!({"anything": "goes"});
    assert!(validate_tool_arguments(&tool, &args).is_ok());
}

#[test]
fn validate_tool_call_finds_tool() {
    let tools = vec![calc_tool()];
    let tc = ToolCall {
        id: "tc-1".to_string(),
        name: "calculate".to_string(),
        arguments: json!({"expression": "2+2"}),
        thought_signature: None,
    };

    let result = validate_tool_call(&tools, &tc);
    assert!(result.is_ok());
    let (tool, validated) = result.unwrap();
    assert_eq!(tool.name, "calculate");
    assert_eq!(validated, json!({"expression": "2+2"}));
}

#[test]
fn validate_tool_call_unknown_tool() {
    let tools = vec![calc_tool()];
    let tc = ToolCall {
        id: "tc-1".to_string(),
        name: "nonexistent".to_string(),
        arguments: json!({}),
        thought_signature: None,
    };

    let result = validate_tool_call(&tools, &tc);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ValidationError::UnknownTool { .. }));
}
