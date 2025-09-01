"""
Agent Communication Protocol - Structured prompts and response parsing.

This module provides standardized prompt injection and response parsing
for agent orchestration with clear communication protocols.
"""

import re
from dataclasses import dataclass

from tools.shared.agent_models import AgentType


@dataclass
class AgentResponse:
    """Parsed structured response from an agent with comprehensive communication support."""
    # Core response fields
    summary: str = ""
    status: str = "completed"  # starting | analyzing | working | testing | blocked | needs_input | completed | failed
    raw_output: str = ""

    # Progress and activity tracking
    progress: str = ""  # "step: 3/7", "65%", etc.
    current_activity: str = ""
    confidence: str = "medium"  # high | medium | low

    # Actions tracking (expanded)
    actions: list[str] = None
    actions_completed: list[str] = None
    actions_in_progress: list[str] = None
    actions_planned: list[str] = None
    actions_blocked: list[str] = None

    # File operations (expanded)
    files_created: list[str] = None
    files_modified: list[str] = None
    files_deleted: list[str] = None
    files_moved: list[str] = None  # Format: "old_path -> new_path"

    # Communication (enhanced)
    questions: list[str] = None
    questions_clarification: list[str] = None
    questions_permission: list[str] = None
    questions_technical: list[str] = None
    questions_preference: list[str] = None

    # Quality and validation
    warnings: list[str] = None
    recommendations: list[str] = None
    observations: list[str] = None
    test_results: dict[str, any] = None
    code_quality: dict[str, any] = None
    validation: dict[str, any] = None

    # Resource monitoring
    resources_used: dict[str, any] = None
    dependencies_added: list[str] = None
    tools_used: list[str] = None

    # Context and metadata
    context: dict[str, any] = None
    environment: dict[str, any] = None
    timestamps: dict[str, str] = None
    session_info: dict[str, str] = None
    metrics: dict[str, any] = None

    def __post_init__(self):
        # Initialize all list and dict fields
        list_fields = [
            'actions', 'actions_completed', 'actions_in_progress', 'actions_planned', 'actions_blocked',
            'files_created', 'files_modified', 'files_deleted', 'files_moved',
            'questions', 'questions_clarification', 'questions_permission', 'questions_technical', 'questions_preference',
            'warnings', 'recommendations', 'observations', 'dependencies_added', 'tools_used'
        ]

        dict_fields = [
            'test_results', 'code_quality', 'validation', 'resources_used',
            'context', 'environment', 'timestamps', 'session_info', 'metrics'
        ]

        for field in list_fields:
            if getattr(self, field) is None:
                setattr(self, field, [])

        for field in dict_fields:
            if getattr(self, field) is None:
                setattr(self, field, {})


class AgentPromptInjector:
    """Injects structured communication prompts into agent messages."""

    # Base communication protocol prompt
    COMMUNICATION_PROTOCOL = """
IMPORTANT: Structure your response using these XML tags for comprehensive communication:

=== CORE STATUS & PROGRESS ===

<STATUS>starting | analyzing | working | testing | blocked | needs_input | completed | failed</STATUS>

<PROGRESS>step: X/Y OR percentage: XX% OR estimate_remaining: Xm XXs</PROGRESS>

<CURRENT_ACTIVITY>What you're currently doing (real-time updates)</CURRENT_ACTIVITY>

<CONFIDENCE>high | medium | low</CONFIDENCE>

<SUMMARY>Provide a 1-2 sentence summary of what you accomplished or attempted.</SUMMARY>

=== ACTIONS & OPERATIONS ===

<ACTIONS_COMPLETED>
- Action description 1
- Action description 2
</ACTIONS_COMPLETED>

<ACTIONS_IN_PROGRESS>
- Currently executing action 1
- Currently executing action 2
</ACTIONS_IN_PROGRESS>

<ACTIONS_PLANNED>
- Next planned action 1
- Next planned action 2
</ACTIONS_PLANNED>

<ACTIONS_BLOCKED>
- Blocked action with reason
</ACTIONS_BLOCKED>

=== FILE & RESOURCE OPERATIONS ===

<FILES_CREATED>
/path/to/new_file.py
/path/to/another_file.js
</FILES_CREATED>

<FILES_MODIFIED>
/path/to/existing_file.py - description of changes
/path/to/another_file.js - description of changes
</FILES_MODIFIED>

<FILES_DELETED>
/path/to/removed_file.py
</FILES_DELETED>

<FILES_MOVED>
/old/path/file.py -> /new/path/file.py
</FILES_MOVED>

<DEPENDENCIES_ADDED>
package_name==version
another_package>=1.0.0
</DEPENDENCIES_ADDED>

<TOOLS_USED>
- git commit -m "message"
- pytest tests/ -v
- npm install package
</TOOLS_USED>

<RESOURCES_USED>
memory: XXMb
cpu: XX%
tokens_consumed: XXXX
api_calls: XX
</RESOURCES_USED>

=== COMMUNICATION & FEEDBACK ===

<QUESTIONS>
  <CLARIFICATION>Specific question about requirements?</CLARIFICATION>
  <PERMISSION>Can I delete/modify this sensitive file?</PERMISSION>
  <TECHNICAL>Which approach do you prefer for X?</TECHNICAL>
  <PREFERENCE>Should I add feature Y or keep it simple?</PREFERENCE>
</QUESTIONS>

<WARNINGS>
- Security concern or risk identified
- Performance issue detected
</WARNINGS>

<RECOMMENDATIONS>
- Suggested improvement or best practice
- Follow-up action recommendation
</RECOMMENDATIONS>

<OBSERVATIONS>
- Notable finding about the codebase
- Pattern or issue discovered
</OBSERVATIONS>

=== QUALITY & VALIDATION ===

<TEST_RESULTS>
passed: XX
failed: XX
coverage: XX%
duration: XXs
</TEST_RESULTS>

<CODE_QUALITY>
complexity_score: X.X/10
maintainability: A/B/C/D
security_issues: XX minor/major
performance_rating: excellent/good/poor
</CODE_QUALITY>

<VALIDATION>
syntax_check: passed/failed
linting: XX warnings
type_check: passed/failed
security_scan: XX issues
</VALIDATION>

=== CONTEXT & METADATA ===

<CONTEXT>
project_type: web_application/cli_tool/library/etc
language: python/javascript/etc
framework: flask/react/etc
environment: development/staging/production
</CONTEXT>

<ENVIRONMENT>
working_directory: /path/to/project
git_branch: branch_name
python_version: X.X.X
node_version: XX.X.X
</ENVIRONMENT>

Always include STATUS, SUMMARY, and any relevant sections. Use empty tags if no content: <TAG></TAG>
Update CURRENT_ACTIVITY and PROGRESS throughout long-running tasks.

---

USER REQUEST:
"""

    # Agent-specific customizations
    AGENT_CUSTOMIZATIONS = {
        AgentType.CLAUDE: {
            "prefix": "You are Claude Code working in an orchestrated environment.",
            "tools_hint": "Use Write, Edit, Read, and Bash tools as needed.",
            "style": "Be thorough but concise in your structured response."
        },
        AgentType.AUGGIE: {
            "prefix": "You are Auggie working in headless --print mode within an orchestration system.",
            "tools_hint": "You can create and modify files using your built-in capabilities.",
            "style": "Focus on clear, actionable outputs in your structured response."
        },
        AgentType.GEMINI: {
            "prefix": "You are Gemini CLI working in --prompt mode within an orchestration system.",
            "tools_hint": "Use your available tools to complete the task effectively.",
            "style": "Provide precise, structured information in your response."
        },
        AgentType.CODEX: {
            "prefix": "You are Codex CLI working in exec mode within an orchestration system.",
            "tools_hint": "Execute tasks using available development tools and commands.",
            "style": "Structure your response to clearly show what was executed."
        }
    }

    @classmethod
    def inject_protocol(cls, message: str, agent_type: AgentType) -> str:
        """Inject communication protocol into agent message."""
        customization = cls.AGENT_CUSTOMIZATIONS.get(agent_type, {})

        # Build the enhanced prompt
        enhanced_prompt = []

        # Add agent-specific prefix
        if "prefix" in customization:
            enhanced_prompt.append(customization["prefix"])

        # Add tools hint
        if "tools_hint" in customization:
            enhanced_prompt.append(customization["tools_hint"])

        # Add style guidance
        if "style" in customization:
            enhanced_prompt.append(customization["style"])

        # Add communication protocol
        enhanced_prompt.append(cls.COMMUNICATION_PROTOCOL)

        # Add the actual user message
        enhanced_prompt.append(message)

        return "\n\n".join(enhanced_prompt)


class AgentResponseParser:
    """Parses structured responses from agents with comprehensive XML tag support."""

    # Comprehensive XML tag patterns
    TAG_PATTERNS = {
        # Core status and progress
        'status': r'<STATUS>(.*?)</STATUS>',
        'summary': r'<SUMMARY>(.*?)</SUMMARY>',
        'progress': r'<PROGRESS>(.*?)</PROGRESS>',
        'current_activity': r'<CURRENT_ACTIVITY>(.*?)</CURRENT_ACTIVITY>',
        'confidence': r'<CONFIDENCE>(.*?)</CONFIDENCE>',

        # Actions (expanded)
        'actions': r'<ACTIONS>(.*?)</ACTIONS>',  # Legacy support
        'actions_completed': r'<ACTIONS_COMPLETED>(.*?)</ACTIONS_COMPLETED>',
        'actions_in_progress': r'<ACTIONS_IN_PROGRESS>(.*?)</ACTIONS_IN_PROGRESS>',
        'actions_planned': r'<ACTIONS_PLANNED>(.*?)</ACTIONS_PLANNED>',
        'actions_blocked': r'<ACTIONS_BLOCKED>(.*?)</ACTIONS_BLOCKED>',

        # File operations (expanded)
        'files_created': r'<FILES_CREATED>(.*?)</FILES_CREATED>',
        'files_modified': r'<FILES_MODIFIED>(.*?)</FILES_MODIFIED>',
        'files_deleted': r'<FILES_DELETED>(.*?)</FILES_DELETED>',
        'files_moved': r'<FILES_MOVED>(.*?)</FILES_MOVED>',

        # Communication (enhanced)
        'questions': r'<QUESTIONS>(.*?)</QUESTIONS>',  # Legacy support
        'questions_clarification': r'<CLARIFICATION>(.*?)</CLARIFICATION>',
        'questions_permission': r'<PERMISSION>(.*?)</PERMISSION>',
        'questions_technical': r'<TECHNICAL>(.*?)</TECHNICAL>',
        'questions_preference': r'<PREFERENCE>(.*?)</PREFERENCE>',

        # Quality and feedback
        'warnings': r'<WARNINGS>(.*?)</WARNINGS>',
        'recommendations': r'<RECOMMENDATIONS>(.*?)</RECOMMENDATIONS>',
        'observations': r'<OBSERVATIONS>(.*?)</OBSERVATIONS>',
        'test_results': r'<TEST_RESULTS>(.*?)</TEST_RESULTS>',
        'code_quality': r'<CODE_QUALITY>(.*?)</CODE_QUALITY>',
        'validation': r'<VALIDATION>(.*?)</VALIDATION>',

        # Resource monitoring
        'resources_used': r'<RESOURCES_USED>(.*?)</RESOURCES_USED>',
        'dependencies_added': r'<DEPENDENCIES_ADDED>(.*?)</DEPENDENCIES_ADDED>',
        'tools_used': r'<TOOLS_USED>(.*?)</TOOLS_USED>',

        # Context and metadata
        'context': r'<CONTEXT>(.*?)</CONTEXT>',
        'environment': r'<ENVIRONMENT>(.*?)</ENVIRONMENT>',
        'timestamps': r'<TIMESTAMPS>(.*?)</TIMESTAMPS>',
        'session_info': r'<SESSION>(.*?)</SESSION>',
        'metrics': r'<METRICS>(.*?)</METRICS>'
    }

    @classmethod
    def parse_response(cls, raw_output: str) -> AgentResponse:
        """Parse comprehensive structured agent response."""
        response = AgentResponse(raw_output=raw_output)

        # Extract each section using regex
        for field, pattern in cls.TAG_PATTERNS.items():
            match = re.search(pattern, raw_output, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()

                if not content:  # Skip empty tags
                    continue

                # String fields
                if field in ['summary', 'status', 'progress', 'current_activity', 'confidence']:
                    if field == 'status':
                        setattr(response, field, content.lower())
                    else:
                        setattr(response, field, content)

                # List fields - actions and questions
                elif field in [
                    'actions', 'actions_completed', 'actions_in_progress', 'actions_planned', 'actions_blocked',
                    'questions', 'questions_clarification', 'questions_permission', 'questions_technical', 'questions_preference',
                    'warnings', 'recommendations', 'observations', 'dependencies_added', 'tools_used'
                ]:
                    items = cls._parse_list_content(content)
                    setattr(response, field, items)

                # File operation fields
                elif field in ['files_created', 'files_modified', 'files_deleted', 'files_moved']:
                    files = cls._parse_file_content(content)
                    setattr(response, field, files)

                # Dictionary fields - structured data
                elif field in ['test_results', 'code_quality', 'validation', 'resources_used', 'context', 'environment', 'timestamps', 'session_info', 'metrics']:
                    parsed_dict = cls._parse_dict_content(content)
                    setattr(response, field, parsed_dict)

        # Handle nested QUESTIONS structure
        cls._parse_nested_questions(raw_output, response)

        # Fallback parsing if no structured tags found
        if not response.summary and not any(getattr(response, field) for field in ['actions', 'actions_completed']):
            response = cls._fallback_parse(raw_output, response)

        return response

    @classmethod
    def _parse_list_content(cls, content: str) -> list[str]:
        """Parse list content from XML tags."""
        items = []
        for line in content.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•')):
                items.append(line.lstrip('- •').strip())
            elif line and not line.startswith('<'):  # Avoid XML artifacts
                items.append(line)
        return items

    @classmethod
    def _parse_file_content(cls, content: str) -> list[str]:
        """Parse file paths from XML tags."""
        files = []
        for line in content.split('\n'):
            line = line.strip()
            if line and (line.startswith('/') or '.' in line or '->' in line):
                files.append(line)
        return files

    @classmethod
    def _parse_dict_content(cls, content: str) -> dict[str, any]:
        """Parse dictionary content from XML tags."""
        result = {}
        for line in content.split('\n'):
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                # Try to convert numeric values
                try:
                    if '.' in value:
                        result[key] = float(value)
                    elif value.isdigit():
                        result[key] = int(value)
                    elif value.lower() in ['true', 'false']:
                        result[key] = value.lower() == 'true'
                    else:
                        result[key] = value
                except ValueError:
                    result[key] = value
        return result

    @classmethod
    def _parse_nested_questions(cls, raw_output: str, response: AgentResponse) -> None:
        """Parse nested QUESTIONS structure with categorized sub-tags."""
        questions_match = re.search(r'<QUESTIONS>(.*?)</QUESTIONS>', raw_output, re.DOTALL | re.IGNORECASE)
        if questions_match:
            questions_content = questions_match.group(1)

            # Parse categorized questions
            categories = {
                'questions_clarification': r'<CLARIFICATION>(.*?)</CLARIFICATION>',
                'questions_permission': r'<PERMISSION>(.*?)</PERMISSION>',
                'questions_technical': r'<TECHNICAL>(.*?)</TECHNICAL>',
                'questions_preference': r'<PREFERENCE>(.*?)</PREFERENCE>'
            }

            for field, pattern in categories.items():
                matches = re.findall(pattern, questions_content, re.DOTALL | re.IGNORECASE)
                if matches:
                    questions = [q.strip() for q in matches if q.strip()]
                    current_questions = getattr(response, field, [])
                    current_questions.extend(questions)
                    setattr(response, field, current_questions)

    @classmethod
    def _fallback_parse(cls, raw_output: str, response: AgentResponse) -> AgentResponse:
        """Fallback parsing for unstructured output."""
        # Try to extract summary from first substantial paragraph
        lines = raw_output.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 20 and not line.startswith('#'):
                response.summary = line[:200] + "..." if len(line) > 200 else line
                break

        # Look for file operations in output
        file_patterns = [
            r'[Cc]reated?\s+(?:file\s+)?([/\w\.-]+\.\w+)',
            r'[Ww]rote\s+(?:to\s+)?([/\w\.-]+\.\w+)',
            r'[Mm]odified\s+([/\w\.-]+\.\w+)',
            r'[Ss]aved?\s+(?:to\s+)?([/\w\.-]+\.\w+)'
        ]

        for pattern in file_patterns:
            matches = re.findall(pattern, raw_output, re.IGNORECASE)
            for match in matches:
                if 'creat' in pattern.lower() or 'wrot' in pattern.lower() or 'saved' in pattern.lower():
                    if match not in response.files_created:
                        response.files_created.append(match)
                elif 'modif' in pattern.lower():
                    if match not in response.files_modified:
                        response.files_modified.append(match)

        # Set default status
        if not response.status:
            if 'error' in raw_output.lower() or 'failed' in raw_output.lower():
                response.status = 'failed'
            elif '?' in raw_output or 'should i' in raw_output.lower():
                response.status = 'needs_input'
            else:
                response.status = 'completed'

        return response

    @classmethod
    def format_parsed_response(cls, response: AgentResponse) -> str:
        """Format comprehensive parsed response for display."""
        lines = []

        # Status and progress
        status_icons = {
            "starting": "🚀", "analyzing": "🔍", "working": "⚡", "testing": "🧪",
            "blocked": "⏸️", "needs_input": "❓", "completed": "✅", "failed": "❌"
        }
        status_icon = status_icons.get(response.status, "⚪")

        lines.append(f"{status_icon} **Status**: {response.status}")

        if response.progress:
            lines.append(f"**Progress**: {response.progress}")

        if response.current_activity:
            lines.append(f"**Current Activity**: {response.current_activity}")

        if response.confidence and response.confidence != "medium":
            confidence_icon = {"high": "🎯", "low": "⚠️"}.get(response.confidence, "")
            lines.append(f"**Confidence**: {confidence_icon} {response.confidence}")

        if response.summary:
            lines.append(f"\n**Summary**: {response.summary}")

        # Actions (comprehensive)
        if response.actions_completed or response.actions:
            lines.append("\n**✅ Actions Completed**:")
            for action in (response.actions_completed or response.actions):
                lines.append(f"  • {action}")

        if response.actions_in_progress:
            lines.append("\n**⚡ Currently Working On**:")
            for action in response.actions_in_progress:
                lines.append(f"  • {action}")

        if response.actions_planned:
            lines.append("\n**📋 Planned Next**:")
            for action in response.actions_planned:
                lines.append(f"  • {action}")

        if response.actions_blocked:
            lines.append("\n**⏸️ Blocked Actions**:")
            for action in response.actions_blocked:
                lines.append(f"  • {action}")

        # Files (expanded)
        if response.files_created:
            lines.append("\n**📄 Files Created**:")
            for file_path in response.files_created:
                lines.append(f"  • {file_path}")

        if response.files_modified:
            lines.append("\n**📝 Files Modified**:")
            for file_path in response.files_modified:
                lines.append(f"  • {file_path}")

        if response.files_deleted:
            lines.append("\n**🗑️ Files Deleted**:")
            for file_path in response.files_deleted:
                lines.append(f"  • {file_path}")

        if response.files_moved:
            lines.append("\n**📁 Files Moved**:")
            for file_path in response.files_moved:
                lines.append(f"  • {file_path}")

        # Dependencies and tools
        if response.dependencies_added:
            lines.append("\n**📦 Dependencies Added**:")
            for dep in response.dependencies_added:
                lines.append(f"  • {dep}")

        if response.tools_used:
            lines.append("\n**🔧 Tools Used**:")
            for tool in response.tools_used:
                lines.append(f"  • {tool}")

        # Questions (categorized)
        questions_sections = [
            ("questions_permission", "🚨 **Requires Permission**", "critical"),
            ("questions_clarification", "❓ **Needs Clarification**", "high"),
            ("questions_technical", "🔧 **Technical Decisions**", "medium"),
            ("questions_preference", "⚙️ **Preferences**", "low")
        ]

        for field, title, _priority in questions_sections:
            questions = getattr(response, field, [])
            if questions:
                lines.append(f"\n{title}:")
                for question in questions:
                    lines.append(f"  • {question}")

        # Legacy questions support
        if response.questions:
            lines.append("\n**❓ Questions/Clarifications Needed**:")
            for question in response.questions:
                lines.append(f"  • {question}")

        # Quality feedback
        if response.warnings:
            lines.append("\n**⚠️ Warnings**:")
            for warning in response.warnings:
                lines.append(f"  • {warning}")

        if response.recommendations:
            lines.append("\n**💡 Recommendations**:")
            for rec in response.recommendations:
                lines.append(f"  • {rec}")

        if response.observations:
            lines.append("\n**👀 Observations**:")
            for obs in response.observations:
                lines.append(f"  • {obs}")

        # Test results and quality metrics
        if response.test_results:
            lines.append("\n**🧪 Test Results**:")
            for key, value in response.test_results.items():
                lines.append(f"  • {key}: {value}")

        if response.code_quality:
            lines.append("\n**📊 Code Quality**:")
            for key, value in response.code_quality.items():
                lines.append(f"  • {key}: {value}")

        # Resource usage
        if response.resources_used:
            lines.append("\n**📈 Resource Usage**:")
            for key, value in response.resources_used.items():
                lines.append(f"  • {key}: {value}")

        return "\n".join(lines)


# Utility functions for easy integration
def enhance_agent_message(message: str, agent_type: AgentType) -> str:
    """Quick function to enhance a message with communication protocol."""
    return AgentPromptInjector.inject_protocol(message, agent_type)


def parse_agent_output(output: str) -> AgentResponse:
    """Quick function to parse agent output."""
    return AgentResponseParser.parse_response(output)


def format_agent_summary(output: str) -> str:
    """Quick function to get formatted summary of agent work."""
    response = parse_agent_output(output)
    return AgentResponseParser.format_parsed_response(response)
