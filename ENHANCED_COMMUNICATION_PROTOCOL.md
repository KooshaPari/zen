# Enhanced Agent Communication Protocol

This document describes the new structured communication protocol for agent orchestration, addressing the need for clear, parseable communication between the lead agent and sub-agents.

## Problem Statement

Traditional agent outputs are verbose, unstructured, and mix reasoning with actionable results:

```
I'll help you create a word counter script. Let me think about this...

First, I need to create a function that opens a file and reads its contents.
Then I'll split the text into words and count them.

I'll use Python's built-in functions for this task.

Here's my implementation:

def count_words(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
            words = content.split()
            return len(words)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return 0

if __name__ == "__main__":
    filename = input("Enter the filename: ")
    word_count = count_words(filename)
    print(f"The file contains {word_count} words.")

I've saved this as word_counter.py in your workspace.
```

**Issues:**
- ❌ Mixed reasoning with results
- ❌ Unclear what actions were actually taken
- ❌ Hard to parse programmatically
- ❌ No clear status indication
- ❌ File operations buried in narrative

## Solution: Structured XML Communication Protocol

### Agent Prompt Enhancement

Each agent receives an enhanced prompt with structured response requirements:

```
IMPORTANT: Structure your response using these XML tags for clear communication:

<SUMMARY>
Provide a 1-2 sentence summary of what you accomplished or attempted.
</SUMMARY>

<ACTIONS>
List each major action you took (one per line):
- Created file: filename.ext
- Modified file: filename.ext
- Ran command: command_name
- Analyzed: what_you_analyzed
</ACTIONS>

<FILES_CREATED>
List any new files you created (full paths, one per line):
/path/to/new_file.py
/path/to/another_file.js
</FILES_CREATED>

<FILES_MODIFIED>
List any existing files you modified (full paths, one per line):
/path/to/existing_file.py
/path/to/another_existing_file.js
</FILES_MODIFIED>

<QUESTIONS>
List any questions or clarifications needed (one per line):
- Should I implement feature X differently?
- Do you want me to add tests for Y?
</QUESTIONS>

<STATUS>
One of: completed | needs_input | failed
</STATUS>

---

USER REQUEST: [Original user message]
```

### Agent-Specific Customizations

Different agents receive customized protocol instructions:

- **Claude**: "You are Claude Code working in an orchestrated environment. Use Write, Edit, Read, and Bash tools as needed."
- **Auggie**: "You are Auggie working in headless --print mode within an orchestration system. You can create and modify files using your built-in capabilities."
- **Gemini**: "You are Gemini CLI working in --prompt mode within an orchestration system. Use your available tools to complete the task effectively."

### Enhanced Agent Output

With the structured protocol, agents produce clear, parseable responses:

```xml
<SUMMARY>
Created a Python word counter script that reads text files and counts words with error handling.
</SUMMARY>

<ACTIONS>
- Created file: word_counter.py
- Implemented word counting function with file reading
- Added error handling for missing files
- Included command-line interface
</ACTIONS>

<FILES_CREATED>
/Users/kooshapari/agents/claude/word_counter.py
</FILES_CREATED>

<FILES_MODIFIED>
</FILES_MODIFIED>

<QUESTIONS>
</QUESTIONS>

<STATUS>
completed
</STATUS>
```

### Parsed and Formatted Output

The system parses the XML and presents clean, structured information:

```
✅ Status: completed
Summary: Created a Python word counter script that reads text files and counts words with error handling.

Actions Taken:
  • Created file: word_counter.py
  • Implemented word counting function with file reading
  • Added error handling for missing files
  • Included command-line interface

Files Created:
  📄 /Users/kooshapari/agents/claude/word_counter.py
```

## Technical Implementation

### 1. Prompt Injection (`AgentPromptInjector`)

```python
def inject_protocol(message: str, agent_type: AgentType) -> str:
    """Inject communication protocol into agent message."""
    customization = AGENT_CUSTOMIZATIONS.get(agent_type, {})
    
    enhanced_prompt = []
    enhanced_prompt.append(customization["prefix"])
    enhanced_prompt.append(customization["tools_hint"])
    enhanced_prompt.append(customization["style"])
    enhanced_prompt.append(COMMUNICATION_PROTOCOL)
    enhanced_prompt.append(message)
    
    return "\n\n".join(enhanced_prompt)
```

### 2. Response Parsing (`AgentResponseParser`)

```python
def parse_response(raw_output: str) -> AgentResponse:
    """Parse structured agent response."""
    response = AgentResponse(raw_output=raw_output)
    
    # Extract each section using regex
    for field, pattern in TAG_PATTERNS.items():
        match = re.search(pattern, raw_output, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            # Process based on field type
            
    # Fallback parsing for unstructured output
    if not response.summary:
        response = _fallback_parse(raw_output, response)
        
    return response
```

### 3. Integration Points

- **Agent Adapters**: Automatically inject protocol into all agent messages
- **Post-processing**: Parse structured responses, fall back to heuristics
- **Metrics**: Enhanced tracking of structured vs unstructured responses

## Benefits

### For Lead Agent (Orchestrator)
1. **Clear Status**: Know if task completed, needs input, or failed
2. **Action Tracking**: Understand exactly what the agent did
3. **File Awareness**: Track which files were created/modified
4. **Question Handling**: Capture clarifications needed for follow-up
5. **Programmatic Parsing**: Reliable extraction of structured data

### For Sub-Agents
1. **Clear Expectations**: Know exactly how to format responses
2. **Reduced Ambiguity**: Structured sections prevent confusion
3. **Agent-Specific Guidance**: Customized instructions per agent type
4. **Fallback Support**: Works even if XML tags are missed

### For System Integration
1. **Reliable Parsing**: XML structure + heuristic fallback
2. **Enhanced Metrics**: Track structured response adoption
3. **Better UX**: Clean, formatted output with status icons
4. **Debugging**: Clear visibility into agent actions and status

## Usage Examples

### Simple Task
```python
# Input
message = "Create a hello world script"
enhanced = enhance_agent_message(message, AgentType.CLAUDE)

# Agent receives enhanced prompt with XML protocol
# Agent responds with structured XML
# System parses and formats for display
```

### Complex Orchestration
```python
# Lead agent gets clear structured feedback
response = parse_agent_output(agent_output)
if response.status == "needs_input":
    # Handle questions
    for question in response.questions:
        clarification = get_user_input(question)
        follow_up_message = f"Answer: {clarification}"
elif response.status == "completed":
    # Continue with next step
    next_agent_task = f"Now work with these files: {response.files_created}"
```

## Comparison: Before vs After

| Aspect | Before | After |
|--------|--------|-------|
| **Clarity** | Mixed reasoning + results | Structured sections |
| **Status** | Inferred from context | Explicit status field |
| **Actions** | Buried in narrative | Clear bullet list |
| **Files** | Manual extraction | Explicit file lists |
| **Questions** | Scattered in text | Dedicated section |
| **Parsing** | Regex heuristics only | XML + fallback |
| **UX** | Wall of text | Formatted with icons |

## Future Enhancements

1. **Multi-turn Conversations**: Maintain conversation context with structured updates
2. **Error Recovery**: Structured error reporting with suggested fixes
3. **Progress Updates**: Structured progress reports for long-running tasks
4. **Resource Usage**: Track computational resources and costs
5. **Agent Learning**: Structured feedback to improve agent performance

## Conclusion

The Enhanced Agent Communication Protocol provides a robust foundation for agent orchestration by establishing clear communication standards while maintaining backward compatibility. This enables the lead agent to effectively coordinate sub-agents, understand their work, and make intelligent decisions about task progression.

Key achievements:
- ✅ Structured XML communication with fallback parsing
- ✅ Agent-specific protocol customization
- ✅ Clear status tracking (completed/needs_input/failed)
- ✅ Explicit file and action tracking
- ✅ Enhanced metrics and debugging capabilities
- ✅ Improved user experience with formatted output

This protocol transforms agent orchestration from guesswork to precision, enabling reliable multi-agent workflows for complex development tasks.