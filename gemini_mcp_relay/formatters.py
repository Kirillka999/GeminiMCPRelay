import base64
from google.genai import types

def unsquash_contents(contents: list[types.Content]) -> list[types.Content]:
    """
    Splits squashed contents into separated messages based on part types.
    e.g. A 'model' content with [functionCall, functionResponse, text] 
    will be split into 'model' (functionCall), 'user' (functionResponse), 'model' (text).
    """
    new_contents = []
    for content in contents:
        if not content.parts:
            new_contents.append(content)
            continue
            
        current_parts = []
        base_role = content.role
        current_role = None
        
        for part in content.parts:
            if part.function_call is not None:
                part_role = "model"
            elif part.function_response is not None:
                part_role = "user"
            else:
                part_role = base_role
                
            if current_role is None:
                current_role = part_role
                
            if part_role != current_role:
                new_contents.append(types.Content(role=current_role, parts=current_parts))
                current_parts = []
                current_role = part_role
                
            current_parts.append(part)
            
        if current_parts:
            new_contents.append(types.Content(role=current_role, parts=current_parts))
            
    return new_contents

def parse_request_payload(payload: dict) -> tuple[list[types.Content], types.GenerateContentConfig]:
    """
    Parses the incoming JSON into strict google.genai.types objects 
    using Pydantic validation.
    """
    contents = []
    for c in payload.get("contents", []):
        contents.append(types.Content.model_validate(c))
        
    # Unsquash history from previous proxy responses
    contents = unsquash_contents(contents)
        
    config = None
    if "generationConfig" in payload:
        config = types.GenerateContentConfig.model_validate(payload["generationConfig"])
    else:
        config = types.GenerateContentConfig()
        
    if "systemInstruction" in payload:
        config.system_instruction = types.Content.model_validate(payload["systemInstruction"])
        
    return contents, config

def convert_bytes_to_b64(obj):
    """
    Recursively converts bytes to base64 strings in a dictionary/list.
    Required for serializing pydantic models to JSON when they contain bytes (like thoughtSignature).
    """
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, bytes):
                obj[k] = base64.b64encode(v).decode('utf-8')
            else:
                convert_bytes_to_b64(v)
    elif isinstance(obj, list):
        for item in obj:
            convert_bytes_to_b64(item)
    return obj

def build_squashed_response(accumulated_parts: list[types.Part]) -> dict:
    """
    Builds a single synthetic response combining all parts (function calls, responses, text)
    so the client SDK receives everything in one turn.
    """
    final_response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                index=0,
                finish_reason=types.FinishReason.STOP,
                content=types.Content(
                    role="model",
                    parts=accumulated_parts
                )
            )
        ]
    )
    response_dict = final_response.model_dump(exclude_none=True, by_alias=True)
    return convert_bytes_to_b64(response_dict)

def build_synthetic_chunk(response_parts: list[types.Part]) -> dict:
    """
    Builds a synthetic streaming chunk specifically to push function responses to the SDK.
    """
    synthetic_chunk = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                index=0,
                content=types.Content(
                    role="model",
                    parts=response_parts
                )
            )
        ]
    )
    synthetic_dict = synthetic_chunk.model_dump(exclude_none=True, by_alias=True)
    return convert_bytes_to_b64(synthetic_dict)

def normalize_tool_schema(raw_schema: dict | None) -> dict:
    """
    Normalizes a JSON schema provided by an MCP server to ensure compatibility
    with Google Gemini API requirements. Handles fallback for missing schemas
    and converts ["type", "null"] arrays into nullable flags.

    This logic is copied from the official Gemini CLI:
    https://github.com/google-gemini/gemini-cli.git
    """
    if not raw_schema:
        return {"type": "object", "properties": {}}

    def _transform_schema(obj):
        if not isinstance(obj, dict):
            if isinstance(obj, list):
                return [_transform_schema(item) for item in obj]
            return obj
        
        res = dict(obj)
        
        # Handle type arrays like ["string", "null"] -> "type": "string", "nullable": True
        type_val = res.get("type")
        if isinstance(type_val, list) and len(type_val) == 2:
            try:
                null_idx = type_val.index("null")
                other_val = type_val[1 - null_idx]
                if isinstance(other_val, str):
                    res["type"] = other_val
                    res["nullable"] = True
            except ValueError:
                pass # "null" not in list
                
        for k, v in res.items():
            res[k] = _transform_schema(v)
            
        return res
        
    input_schema = _transform_schema(raw_schema)
    if "type" not in input_schema:
        input_schema["type"] = "object"
        
    return input_schema
