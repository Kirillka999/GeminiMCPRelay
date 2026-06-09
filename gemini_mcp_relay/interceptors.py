from google.genai import types

class ToolInterceptor:
    """
    Base class for intercepting and modifying tool calls during the autonomous generation loop.
    Override these methods to implement custom logic.
    """
    
    async def before_tool_call(self, tool_call: types.FunctionCall) -> types.FunctionCall | dict:
        """
        Called before a tool is executed.
        
        Args:
            tool_call: The tool call requested by the model.
            
        Returns:
            Either a (potentially modified) types.FunctionCall to proceed with the execution,
            or a dictionary representing the mocked result to bypass the actual tool execution.
        """
        return tool_call

    async def after_tool_call(self, tool_call: types.FunctionCall, result: dict) -> dict:
        """
        Called after a tool has been executed but before the result is sent back to the model.
        
        Args:
            tool_call: The original tool call.
            result: The result returned by the tool execution (or the mock from before_tool_call).
            
        Returns:
            The (potentially modified) result dictionary to be sent to the model.
        """
        return result
