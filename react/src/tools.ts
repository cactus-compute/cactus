import type { NativeCompletionResult } from "./NativeCactus";

interface Parameter {
  type: string,
  description: string,
  required?: boolean 
}

interface Tool {
  func: Function,
  description: string,
  parameters: {[key: string]: Parameter},
  required: string[]
}

export class Tools {
  private tools = new Map<string, Tool>();
  
  add(
      func: Function, 
      description: string,
      parameters: {[key: string]: Parameter},
    ) {
      this.tools.set(func.name, { 
        func, 
        description,
        parameters,
        required: Object.entries(parameters)
          .filter(([_, param]) => param.required)
          .map(([key, _]) => key)
      });
      return func;
    }
  
  getSchemas() {
      return Array.from(this.tools.entries()).map(([name, { description, parameters, required }]) => ({
        type: "function",
        function: {
          name,
          description,
          parameters: {
            type: "object",
            properties: parameters,
            required
          }
        }
      }));
    }
  
  async execute(name: string, args: any) {
      const tool = this.tools.get(name);
      if (!tool) throw new Error(`Tool ${name} not found`);
      return await tool.func(...Object.values(args));
  }
}

export async function parseAndExecuteTool(result: NativeCompletionResult, tools: Tools): Promise<{toolCalled: boolean, toolName?: string, toolInput?: any, toolOutput?: any}> {
  if (!result.tool_calls || result.tool_calls.length === 0) {
      return {toolCalled: false};
  }
  
  try {
      const toolCall = result.tool_calls[0];
      if (!toolCall) {
        return {toolCalled: false};
      }
      const toolName = toolCall.function.name;
      const toolInput = JSON.parse(toolCall.function.arguments);
      
      const toolOutput = await tools.execute(toolName, toolInput);
      
      return {
          toolCalled: true,
          toolName,
          toolInput,
          toolOutput
      };
  } catch (error) {
      return {toolCalled: false};
  }
}