use crate::types::{Tool, ToolCall};

/// Error returned when tool argument validation fails.
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("unknown tool: {name}")]
    UnknownTool { name: String },

    #[error("invalid arguments for tool '{tool}': {message}")]
    InvalidArguments { tool: String, message: String },

    #[error("schema error: {0}")]
    Schema(String),
}

/// Validate that `arguments` conform to the tool's JSON Schema parameters.
///
/// Returns the validated (possibly coerced) arguments on success.
pub fn validate_tool_arguments(
    tool: &Tool,
    arguments: &serde_json::Value,
) -> Result<serde_json::Value, ValidationError> {
    let schema = &tool.parameters;

    // If the schema is an empty object or boolean `true`, anything is valid.
    if schema.is_boolean() || (schema.is_object() && schema.as_object().unwrap().is_empty()) {
        return Ok(arguments.clone());
    }

    if let Err(e) = jsonschema::validate(schema, arguments) {
        return Err(ValidationError::InvalidArguments {
            tool: tool.name.clone(),
            message: e.to_string(),
        });
    }

    Ok(arguments.clone())
}

/// Find the matching tool for a tool call, then validate the call's arguments.
pub fn validate_tool_call(
    tools: &[Tool],
    tool_call: &ToolCall,
) -> Result<(Tool, serde_json::Value), ValidationError> {
    let tool = tools
        .iter()
        .find(|t| t.name == tool_call.name)
        .ok_or_else(|| ValidationError::UnknownTool {
            name: tool_call.name.clone(),
        })?;

    let validated = validate_tool_arguments(tool, &tool_call.arguments)?;
    Ok((tool.clone(), validated))
}
