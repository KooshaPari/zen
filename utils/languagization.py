"""
Output Languagization System for Natural Progress Feeds.

This module converts structured agent responses and status updates into natural,
conversational language that provides intuitive understanding of agent progress
and activities.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from utils.agent_prompts import AgentResponse
from utils.streaming_protocol import StreamMessage, StreamMessageType

logger = logging.getLogger(__name__)


class NarrativeStyle(Enum):
    """Different narrative styles for progress updates."""
    CONVERSATIONAL = "conversational"  # Natural, friendly tone
    TECHNICAL = "technical"           # Precise, developer-focused
    EXECUTIVE = "executive"           # High-level, business-focused
    DETAILED = "detailed"            # Comprehensive, step-by-step


@dataclass
class ProgressNarrative:
    """Natural language narrative of agent progress."""
    current_status: str
    activity_description: str
    progress_summary: str
    time_estimate: Optional[str] = None
    confidence_level: str = "medium"
    next_steps: list[str] = None
    concerns: list[str] = None
    achievements: list[str] = None

    def __post_init__(self):
        if self.next_steps is None:
            self.next_steps = []
        if self.concerns is None:
            self.concerns = []
        if self.achievements is None:
            self.achievements = []


class StatusLanguagizer:
    """Converts status updates into natural language narratives."""

    def __init__(self):
        self.status_phrases = {
            "starting": [
                "🚀 Getting started",
                "🚀 Beginning work on",
                "🚀 Initializing",
                "🚀 Starting up"
            ],
            "analyzing": [
                "🔍 Taking a closer look at",
                "🔍 Analyzing the situation",
                "🔍 Examining",
                "🔍 Investigating"
            ],
            "working": [
                "⚡ Making progress on",
                "⚡ Working hard on",
                "⚡ Actively developing",
                "⚡ Building"
            ],
            "testing": [
                "🧪 Testing the changes",
                "🧪 Running validation",
                "🧪 Checking everything works",
                "🧪 Verifying functionality"
            ],
            "blocked": [
                "⏸️ Hit a roadblock with",
                "⏸️ Encountered an issue",
                "⏸️ Need help with",
                "⏸️ Waiting for clarification on"
            ],
            "needs_input": [
                "❓ Need your input on",
                "❓ Waiting for guidance about",
                "❓ Have a question about",
                "❓ Need clarification on"
            ],
            "completed": [
                "✅ Successfully finished",
                "✅ Completed work on",
                "✅ All done with",
                "✅ Mission accomplished for"
            ],
            "failed": [
                "❌ Ran into problems with",
                "❌ Couldn't complete",
                "❌ Failed to finish",
                "❌ Encountered errors with"
            ]
        }

        self.progress_descriptions = {
            "just_started": "just getting started",
            "early_stages": "in the early stages",
            "making_headway": "making good headway",
            "halfway_there": "about halfway through",
            "well_underway": "well underway",
            "nearly_done": "nearly finished",
            "finishing_up": "putting on the finishing touches",
            "completed": "completed successfully"
        }

        self.confidence_expressions = {
            "high": ["confident", "sure", "on track", "going smoothly"],
            "medium": ["proceeding normally", "making steady progress", "things are going well"],
            "low": ["uncertain", "might need help", "encountering some challenges", "working through issues"]
        }

    def interpret_status_progression(self, status_history: list[dict[str, Any]]) -> str:
        """Convert status updates into natural language narrative."""
        if not status_history:
            return "No updates available yet."

        narratives = []

        for i, update in enumerate(status_history):
            status = update.get("status", "unknown")
            activity = update.get("current_activity", "")
            progress = update.get("progress", "")
            confidence = update.get("confidence", "medium")

            if i == 0:
                # First update
                phrase = self._random_choice(self.status_phrases.get(status, ["Working on"]))
                if activity:
                    narratives.append(f"{phrase} {activity.lower()}")
                else:
                    narratives.append(f"{phrase} the task")

            elif status == "blocked":
                activity_desc = activity or "the current step"
                narratives.append(f"⏸️ Hit a roadblock: {activity_desc}")

            elif status == "working":
                progress_phrase = self.progress_to_language(progress)
                activity_desc = activity or "the task"
                confidence_phrase = self._get_confidence_phrase(confidence)
                narratives.append(f"⚡ Making progress: {activity_desc} ({progress_phrase}) - {confidence_phrase}")

            elif status == "completed":
                summary = self._summarize_accomplishments(update)
                narratives.append(f"✅ Finished successfully: {summary}")

        return "\n".join(narratives)

    def progress_to_language(self, progress: str) -> str:
        """Convert progress indicators to natural language."""
        if not progress:
            return "in progress"

        progress = progress.lower().strip()

        # Handle percentage progress
        if "%" in progress:
            try:
                pct = int(re.findall(r'\d+', progress)[0])
                if pct < 10:
                    return "just getting started"
                elif pct < 25:
                    return "in the early stages"
                elif pct < 50:
                    return "making good headway"
                elif pct < 75:
                    return "well underway"
                elif pct < 95:
                    return "nearly finished"
                else:
                    return "putting on the finishing touches"
            except (ValueError, IndexError):
                pass

        # Handle step progress
        if "step:" in progress:
            try:
                parts = progress.replace("step:", "").strip().split("/")
                if len(parts) == 2:
                    current, total = int(parts[0]), int(parts[1])
                    percentage = (current / total) * 100
                    return self.progress_to_language(f"{percentage}%")
            except (ValueError, IndexError):
                pass

        # Handle time estimates
        if any(unit in progress for unit in ["minute", "min", "hour", "second", "sec"]):
            return f"about {progress} remaining"

        return progress

    def _random_choice(self, options: list[str]) -> str:
        """Select a random phrase from options."""
        import random
        return random.choice(options) if options else "Working on"

    def _get_confidence_phrase(self, confidence: str) -> str:
        """Get a natural phrase for confidence level."""
        phrases = self.confidence_expressions.get(confidence, ["proceeding"])
        return self._random_choice(phrases)

    def _summarize_accomplishments(self, update: dict[str, Any]) -> str:
        """Summarize what was accomplished."""
        # Extract key accomplishments from the update
        files_created = update.get("files_created", [])
        files_modified = update.get("files_modified", [])
        actions = update.get("actions_completed", [])

        accomplishments = []

        if files_created:
            accomplishments.append(f"created {len(files_created)} files")
        if files_modified:
            accomplishments.append(f"updated {len(files_modified)} files")
        if actions:
            accomplishments.append(f"completed {len(actions)} tasks")

        if accomplishments:
            if len(accomplishments) == 1:
                return accomplishments[0]
            elif len(accomplishments) == 2:
                return f"{accomplishments[0]} and {accomplishments[1]}"
            else:
                return f"{', '.join(accomplishments[:-1])}, and {accomplishments[-1]}"

        return "the assigned work"


class ActionLanguagizer:
    """Converts action lists into coherent narratives."""

    def __init__(self):
        self.action_categories = {
            "file_operations": {
                "patterns": ["created", "modified", "deleted", "moved", "saved", "wrote"],
                "summary": "file operations"
            },
            "code_changes": {
                "patterns": ["implemented", "added", "fixed", "refactored", "optimized", "updated"],
                "summary": "code improvements"
            },
            "testing": {
                "patterns": ["tested", "validated", "verified", "checked", "ran tests"],
                "summary": "testing activities"
            },
            "configuration": {
                "patterns": ["configured", "setup", "installed", "initialized", "deployed"],
                "summary": "configuration changes"
            },
            "analysis": {
                "patterns": ["analyzed", "reviewed", "examined", "investigated", "researched"],
                "summary": "analysis work"
            }
        }

    def summarize_actions(self, actions: list[str], style: NarrativeStyle = NarrativeStyle.CONVERSATIONAL) -> str:
        """Convert action lists into coherent narrative."""
        if not actions:
            return "No major actions taken yet."

        # Categorize actions
        categorized = self._categorize_actions(actions)

        if style == NarrativeStyle.CONVERSATIONAL:
            return self._conversational_summary(categorized)
        elif style == NarrativeStyle.TECHNICAL:
            return self._technical_summary(categorized)
        elif style == NarrativeStyle.EXECUTIVE:
            return self._executive_summary(categorized)
        else:
            return self._detailed_summary(categorized, actions)

    def _categorize_actions(self, actions: list[str]) -> dict[str, list[str]]:
        """Categorize actions by type."""
        categorized = {category: [] for category in self.action_categories}
        uncategorized = []

        for action in actions:
            action_lower = action.lower()
            matched = False

            for category, config in self.action_categories.items():
                if any(pattern in action_lower for pattern in config["patterns"]):
                    categorized[category].append(action)
                    matched = True
                    break

            if not matched:
                uncategorized.append(action)

        if uncategorized:
            categorized["other"] = uncategorized

        # Remove empty categories
        return {k: v for k, v in categorized.items() if v}

    def _conversational_summary(self, categorized: dict[str, list[str]]) -> str:
        """Create conversational summary of actions."""
        summaries = []

        for category, actions in categorized.items():
            if category == "file_operations":
                file_count = len([a for a in actions if any(word in a.lower() for word in ["created", "saved", "wrote"])])
                if file_count > 0:
                    summaries.append(f"worked with {file_count} files")
            elif category == "code_changes":
                summaries.append(f"made {len(actions)} code improvements")
            elif category == "testing":
                summaries.append(f"ran {len(actions)} tests")
            elif category == "configuration":
                summaries.append(f"handled {len(actions)} configuration tasks")
            elif category == "analysis":
                summaries.append("performed analysis work")
            else:
                summaries.append(f"completed {len(actions)} other tasks")

        return self._join_naturally(summaries)

    def _technical_summary(self, categorized: dict[str, list[str]]) -> str:
        """Create technical summary of actions."""
        summaries = []

        for category, actions in categorized.items():
            if category in self.action_categories:
                category_name = self.action_categories[category]["summary"]
                summaries.append(f"{len(actions)} {category_name}")
            else:
                summaries.append(f"{len(actions)} other actions")

        return f"Completed: {', '.join(summaries)}"

    def _executive_summary(self, categorized: dict[str, list[str]]) -> str:
        """Create executive summary of actions."""
        total_actions = sum(len(actions) for actions in categorized.values())
        key_areas = []

        if "code_changes" in categorized:
            key_areas.append("development work")
        if "file_operations" in categorized:
            key_areas.append("file management")
        if "testing" in categorized:
            key_areas.append("quality assurance")
        if "configuration" in categorized:
            key_areas.append("system configuration")

        areas_text = self._join_naturally(key_areas) if key_areas else "various tasks"
        return f"Completed {total_actions} actions across {areas_text}"

    def _detailed_summary(self, categorized: dict[str, list[str]], all_actions: list[str]) -> str:
        """Create detailed summary with specific actions."""
        lines = []

        for category, actions in categorized.items():
            if category in self.action_categories:
                category_name = self.action_categories[category]["summary"].title()
            else:
                category_name = "Other Actions"

            lines.append(f"**{category_name}:**")
            for action in actions:
                lines.append(f"  • {action}")

        return "\n".join(lines)

    def _join_naturally(self, items: list[str]) -> str:
        """Join a list of items with natural language connectors."""
        if not items:
            return "nothing"
        elif len(items) == 1:
            return items[0]
        elif len(items) == 2:
            return f"{items[0]} and {items[1]}"
        else:
            return f"{', '.join(items[:-1])}, and {items[-1]}"


class QuestionLanguagizer:
    """Formats questions by category with appropriate urgency and context."""

    def __init__(self):
        self.priority_levels = {
            "permission": {"icon": "🚨", "urgency": "critical", "intro": "I need your permission to"},
            "clarification": {"icon": "❓", "urgency": "high", "intro": "I need clarification on"},
            "technical": {"icon": "🔧", "urgency": "medium", "intro": "I have a technical question about"},
            "preference": {"icon": "⚙️", "urgency": "low", "intro": "What would you prefer for"}
        }

    def format_questions(
        self,
        questions: dict[str, list[str]],
        style: NarrativeStyle = NarrativeStyle.CONVERSATIONAL
    ) -> str:
        """Format questions by category with appropriate urgency and natural language."""
        if not any(questions.values()):
            return "No questions at this time."

        formatted_sections = []

        # Order by priority (permission first, preference last)
        priority_order = ["permission", "clarification", "technical", "preference"]

        for category in priority_order:
            category_questions = questions.get(f"questions_{category}", [])
            if not category_questions:
                continue

            config = self.priority_levels[category]

            if style == NarrativeStyle.CONVERSATIONAL:
                formatted_sections.append(self._conversational_questions(category, category_questions, config))
            elif style == NarrativeStyle.TECHNICAL:
                formatted_sections.append(self._technical_questions(category, category_questions, config))
            else:
                formatted_sections.append(self._detailed_questions(category, category_questions, config))

        return "\n\n".join(formatted_sections)

    def _conversational_questions(self, category: str, questions: list[str], config: dict[str, str]) -> str:
        """Format questions in conversational style."""
        lines = []
        icon = config["icon"]
        intro = config["intro"]

        if len(questions) == 1:
            lines.append(f"{icon} {intro} {questions[0].lower()}")
        else:
            lines.append(f"{icon} {intro}:")
            for q in questions:
                lines.append(f"  • {q}")

        return "\n".join(lines)

    def _technical_questions(self, category: str, questions: list[str], config: dict[str, str]) -> str:
        """Format questions in technical style."""
        lines = []
        icon = config["icon"]
        category_title = category.replace("_", " ").title()

        lines.append(f"{icon} **{category_title} ({config['urgency']} priority):**")
        for q in questions:
            lines.append(f"  • {q}")

        return "\n".join(lines)

    def _detailed_questions(self, category: str, questions: list[str], config: dict[str, str]) -> str:
        """Format questions with detailed context."""
        lines = []
        icon = config["icon"]
        category_title = category.replace("_", " ").title()

        lines.append(f"{icon} **{category_title} Questions** (Priority: {config['urgency']})")
        lines.append(f"*{config['intro']}:*")

        for i, q in enumerate(questions, 1):
            lines.append(f"{i}. {q}")

        return "\n".join(lines)


class ProgressFeedGenerator:
    """Generates natural language progress feeds from streaming updates."""

    def __init__(self, style: NarrativeStyle = NarrativeStyle.CONVERSATIONAL):
        self.style = style
        self.status_languagizer = StatusLanguagizer()
        self.action_languagizer = ActionLanguagizer()
        self.question_languagizer = QuestionLanguagizer()

        self.task_histories: dict[str, list[dict[str, Any]]] = {}

    def process_stream_message(self, message: StreamMessage) -> Optional[str]:
        """Process a streaming message and generate natural language update."""
        task_id = message.task_id

        # Initialize history for new tasks
        if task_id not in self.task_histories:
            self.task_histories[task_id] = []

        # Convert stream message to update format
        update = {
            "timestamp": message.timestamp,
            "type": message.type.value,
            "content": message.content,
            "sequence": message.sequence
        }

        # Generate natural language based on message type
        if message.type == StreamMessageType.STATUS_UPDATE:
            return self._generate_status_narrative(task_id, update)
        elif message.type == StreamMessageType.PROGRESS_UPDATE:
            return self._generate_progress_narrative(task_id, update)
        elif message.type == StreamMessageType.ACTIVITY_UPDATE:
            return self._generate_activity_narrative(task_id, update)
        elif message.type == StreamMessageType.ACTION_UPDATE:
            return self._generate_action_narrative(task_id, update)
        elif message.type == StreamMessageType.QUESTION_UPDATE:
            return self._generate_question_narrative(task_id, update)
        elif message.type == StreamMessageType.WARNING:
            return self._generate_warning_narrative(task_id, update)
        elif message.type == StreamMessageType.COMPLETION:
            return self._generate_completion_narrative(task_id, update)

        return None

    def _generate_status_narrative(self, task_id: str, update: dict[str, Any]) -> str:
        """Generate natural language for status updates."""
        content = update["content"]
        status = content.get("status", "unknown")
        message = content.get("message", "")

        phrases = self.status_languagizer.status_phrases.get(status, ["Working on"])
        phrase = self.status_languagizer._random_choice(phrases)

        if message:
            return f"{phrase}: {message.lower()}"
        else:
            return f"{phrase} the task"

    def _generate_progress_narrative(self, task_id: str, update: dict[str, Any]) -> str:
        """Generate natural language for progress updates."""
        content = update["content"]
        progress = content.get("content", "")

        progress_desc = self.status_languagizer.progress_to_language(progress)
        return f"📈 Progress update: {progress_desc}"

    def _generate_activity_narrative(self, task_id: str, update: dict[str, Any]) -> str:
        """Generate natural language for activity updates."""
        content = update["content"]
        activity = content.get("content", "")

        return f"⚡ Now working on: {activity.lower()}"

    def _generate_action_narrative(self, task_id: str, update: dict[str, Any]) -> str:
        """Generate natural language for action updates."""
        content = update["content"]
        tag = content.get("tag", "")
        actions_text = content.get("content", "")

        # Parse actions from text
        actions = [line.strip().lstrip('- •') for line in actions_text.split('\n') if line.strip()]

        if tag == "ACTIONS_COMPLETED":
            summary = self.action_languagizer.summarize_actions(actions, self.style)
            return f"✅ Just completed: {summary}"
        elif tag == "ACTIONS_IN_PROGRESS":
            if len(actions) == 1:
                return f"⚡ Currently: {actions[0].lower()}"
            else:
                return f"⚡ Working on {len(actions)} tasks simultaneously"

        return f"📋 Action update: {len(actions)} items"

    def _generate_question_narrative(self, task_id: str, update: dict[str, Any]) -> str:
        """Generate natural language for questions."""
        content = update["content"]
        questions_text = content.get("content", "")

        # Simple question parsing
        questions = [line.strip() for line in questions_text.split('\n') if line.strip()]

        if len(questions) == 1:
            return f"❓ Quick question: {questions[0]}"
        else:
            return f"❓ Have {len(questions)} questions that need your input"

    def _generate_warning_narrative(self, task_id: str, update: dict[str, Any]) -> str:
        """Generate natural language for warnings."""
        content = update["content"]
        warnings_text = content.get("content", "")

        warnings = [line.strip().lstrip('- •') for line in warnings_text.split('\n') if line.strip()]

        if len(warnings) == 1:
            return f"⚠️ Heads up: {warnings[0].lower()}"
        else:
            return f"⚠️ Found {len(warnings)} issues that need attention"

    def _generate_completion_narrative(self, task_id: str, update: dict[str, Any]) -> str:
        """Generate natural language for task completion."""
        content = update["content"]
        status = content.get("status", "completed")
        summary = content.get("summary", "")
        files_created = content.get("files_created", [])
        files_modified = content.get("files_modified", [])

        accomplishments = []
        if files_created:
            accomplishments.append(f"created {len(files_created)} files")
        if files_modified:
            accomplishments.append(f"updated {len(files_modified)} files")

        if status == "completed":
            if accomplishments:
                acc_text = self.action_languagizer._join_naturally(accomplishments)
                return f"✅ All done! {summary or 'Successfully completed the task'} - {acc_text}"
            else:
                return f"✅ All done! {summary or 'Task completed successfully'}"
        elif status == "failed":
            return f"❌ Task couldn't be completed: {summary or 'Encountered errors'}"
        else:
            return f"⏸️ Task paused: {summary or 'Waiting for input'}"

    def get_task_summary(self, task_id: str) -> str:
        """Get a natural language summary of the entire task progress."""
        if task_id not in self.task_histories:
            return "No progress information available for this task."

        history = self.task_histories[task_id]
        if not history:
            return "Task started but no updates yet."

        # Generate summary based on entire history
        status_updates = [update for update in history if update["type"] == "status_update"]
        return self.status_languagizer.interpret_status_progression(status_updates)


# Global progress feed generator
_progress_generator: Optional[ProgressFeedGenerator] = None


def get_progress_generator(style: NarrativeStyle = NarrativeStyle.CONVERSATIONAL) -> ProgressFeedGenerator:
    """Get the global progress feed generator."""
    global _progress_generator
    if _progress_generator is None or _progress_generator.style != style:
        _progress_generator = ProgressFeedGenerator(style)
    return _progress_generator


def create_natural_progress_feed(agent_response: AgentResponse, style: NarrativeStyle = NarrativeStyle.CONVERSATIONAL) -> str:
    """Create a natural language progress feed from an agent response."""
    generator = get_progress_generator(style)

    # Convert AgentResponse to narrative format
    narratives = []

    # Status and activity
    if agent_response.current_activity:
        narratives.append(f"⚡ {agent_response.current_activity}")

    # Progress
    if agent_response.progress:
        progress_desc = generator.status_languagizer.progress_to_language(agent_response.progress)
        narratives.append(f"📈 Progress: {progress_desc}")

    # Actions
    if agent_response.actions_completed:
        action_summary = generator.action_languagizer.summarize_actions(agent_response.actions_completed, style)
        narratives.append(f"✅ Completed: {action_summary}")

    # Questions
    questions_dict = {
        "questions_permission": agent_response.questions_permission,
        "questions_clarification": agent_response.questions_clarification,
        "questions_technical": agent_response.questions_technical,
        "questions_preference": agent_response.questions_preference
    }

    if any(questions_dict.values()):
        question_text = generator.question_languagizer.format_questions(questions_dict, style)
        narratives.append(question_text)

    # Warnings
    if agent_response.warnings:
        if len(agent_response.warnings) == 1:
            narratives.append(f"⚠️ Heads up: {agent_response.warnings[0]}")
        else:
            narratives.append(f"⚠️ {len(agent_response.warnings)} warnings need attention")

    return "\n\n".join(narratives) if narratives else "Agent is working..."
