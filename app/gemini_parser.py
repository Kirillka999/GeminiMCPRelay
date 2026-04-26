from google.genai import types

def parse_request_payload(payload: dict) -> tuple[list[types.Content], types.GenerateContentConfig]:
    """
    Parses the incoming JSON into strict google.genai.types objects 
    using Pydantic validation.
    """
    contents = []
    for c in payload.get("contents", []):
        contents.append(types.Content.model_validate(c))
        
    config = None
    if "generationConfig" in payload:
        config = types.GenerateContentConfig.model_validate(payload["generationConfig"])
    else:
        config = types.GenerateContentConfig()
        
    if "systemInstruction" in payload:
        config.system_instruction = types.Content.model_validate(payload["systemInstruction"])
        
    return contents, config
