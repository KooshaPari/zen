# Comprehensive Communication Protocol Plan
Status: Plan/spec — see `/ENHANCED_COMMUNICATION_PROTOCOL.md` and `utils/agent_prompts.py` for implemented parts.

Implementation Status (Per Feature)
- Full tag taxonomy (STATUS/PROGRESS/CURRENT_ACTIVITY/CONFIDENCE/etc.): Partially available
  - Implemented in: `utils/agent_prompts.py` (prompt injection + parsing), `/ENHANCED_COMMUNICATION_PROTOCOL.md`
- File operation tracking (FILES_CREATED/MODIFIED/DELETED/MOVED): Partially available
  - Implemented in: `utils/agent_prompts.py` (parsing + fallback heuristics)
- Planned/blocked actions sections: Partially available
  - Implemented in: `utils/agent_prompts.py`
- Resource-aware interrupts (token/time limits): Deferred
  - Nearest modules: `utils/streaming_monitor.py`, `utils/token_budget_manager.py`
- Provenance/compliance signatures: Deferred
  - Nearest modules: `utils/audit_trail.py`, `utils/kafka_events.py`
## Advanced Tag System & Input/Output Transformation Architecture

This document outlines a complete communication protocol with comprehensive tag types, transforms, and languagization for intuitive agent orchestration.

---

## 🏗️ **Tag Taxonomy & Parsing System**

### **Core Status & Progress Tags**
```xml
<!-- Primary Status - Dynamic Updates -->
<STATUS>starting | analyzing | working | testing | blocked | needs_input | completed | failed</STATUS>

<!-- Progress Tracking - Percentage or Step-based -->
<PROGRESS>
  step: 3/7
  percentage: 42%
  estimate_remaining: 2m 30s
</PROGRESS>

<!-- Current Activity - Real-time updates -->
<CURRENT_ACTIVITY>
  Reading requirements.txt to understand dependencies
</CURRENT_ACTIVITY>

<!-- Confidence Level -->
<CONFIDENCE>high | medium | low</CONFIDENCE>
```

### **Action & Operation Tags**
```xml
<!-- Completed Actions -->
<ACTIONS_COMPLETED>
- Analyzed codebase structure
- Created database schema
- Implemented user authentication
- Added error handling
</ACTIONS_COMPLETED>

<!-- Currently Executing -->
<ACTIONS_IN_PROGRESS>
- Running test suite (test_user_auth.py)
- Validating database connections
</ACTIONS_IN_PROGRESS>

<!-- Planned Next Steps -->
<ACTIONS_PLANNED>
- Deploy to staging environment
- Run integration tests
- Update documentation
</ACTIONS_PLANNED>

<!-- Blocked Actions -->
<ACTIONS_BLOCKED>
- Cannot deploy without API keys
- Tests failing due to missing test data
</ACTIONS_BLOCKED>
```

### **File & Resource Management Tags**
```xml
<!-- File Operations -->
<FILES_CREATED>
/src/auth/user_model.py
/tests/test_auth.py
/config/database.yml
</FILES_CREATED>

<FILES_MODIFIED>
/src/main.py - Added authentication middleware
/requirements.txt - Added bcrypt dependency
</FILES_MODIFIED>

<FILES_DELETED>
/old_scripts/legacy_auth.py
</FILES_DELETED>

<FILES_MOVED>
/utils/helpers.py -> /src/utils/auth_helpers.py
</FILES_MOVED>

<!-- Resource Usage -->
<RESOURCES_USED>
memory: 245MB
cpu: 15%
disk_space: 1.2GB
api_calls: 12
tokens_consumed: 2,450
</RESOURCES_USED>

<!-- Dependencies -->
<DEPENDENCIES_ADDED>
bcrypt==4.0.1
pytest==7.4.0
sqlalchemy==2.0.19
</DEPENDENCIES_ADDED>
```

### **Communication & Interaction Tags**
```xml
<!-- Questions - Categorized -->
<QUESTIONS>
  <CLARIFICATION>Should I use JWT tokens or session-based auth?</CLARIFICATION>
  <PERMISSION>Can I delete the old authentication files?</PERMISSION>
  <TECHNICAL>Which database migration strategy do you prefer?</TECHNICAL>
  <PREFERENCE>Do you want me to add logging to all endpoints?</PREFERENCE>
</QUESTIONS>

<!-- Warnings & Concerns -->
<WARNINGS>
- Current password hashing is weak (MD5)
- No rate limiting on login endpoint
- Database credentials in plain text
</WARNINGS>

<!-- Recommendations -->
<RECOMMENDATIONS>
- Implement password complexity requirements
- Add 2FA support for admin accounts
- Set up automated backup system
</RECOMMENDATIONS>

<!-- Observations -->
<OBSERVATIONS>
- Codebase follows PEP 8 standards well
- Good test coverage (85%) in existing modules
- Documentation is minimal but consistent
</OBSERVATIONS>
```

### **Context & Environment Tags**
```xml
<!-- Working Context -->
<CONTEXT>
project_type: web_application
language: python
framework: flask
database: postgresql
environment: development
</CONTEXT>

<!-- Tools & Commands Used -->
<TOOLS_USED>
- git commit -m "Add user authentication"
- pytest tests/test_auth.py -v
- pip install bcrypt
- docker-compose up -d postgres
</TOOLS_USED>

<!-- Environment State -->
<ENVIRONMENT>
working_directory: /Users/dev/project
git_branch: feature/auth-system
virtual_env: .venv
python_version: 3.11.4
</ENVIRONMENT>
```

### **Quality & Testing Tags**
```xml
<!-- Test Results -->
<TEST_RESULTS>
passed: 24
failed: 2
skipped: 1
coverage: 87%
duration: 12.3s
</TEST_RESULTS>

<!-- Code Quality -->
<CODE_QUALITY>
complexity_score: 6.2/10
maintainability: B+
security_issues: 3 minor
performance_rating: good
</CODE_QUALITY>

<!-- Validation -->
<VALIDATION>
syntax_check: passed
linting: 2 warnings
type_check: passed
security_scan: 3 low-priority issues
</VALIDATION>
```

### **Metadata & Tracking Tags**
```xml
<!-- Timestamps -->
<TIMESTAMPS>
started_at: 2025-01-15T14:30:22Z
last_update: 2025-01-15T14:35:18Z
estimated_completion: 2025-01-15T14:45:00Z
</TIMESTAMPS>

<!-- Session Info -->
<SESSION>
task_id: auth-impl-001
session_id: sess_9k2j1m
agent_version: claude-3.5
parent_task: user-management-system
</SESSION>

<!-- Metrics -->
<METRICS>
lines_of_code_added: 324
lines_of_code_modified: 89
files_touched: 12
git_commits: 3
time_spent: 18m 42s
</METRICS>
```

---

## 🔄 **Input Transformation Pipeline**

### **1. Intent Recognition & Preprocessing**
```python
class InputTransformer:
    def preprocess_user_input(self, raw_input: str) -> EnhancedInput:
        """Transform user input into structured agent instructions."""
        
        # Extract intent and priority
        intent = self.extract_intent(raw_input)
        priority = self.extract_priority_signals(raw_input)
        context = self.extract_context_hints(raw_input)
        
        # Parse embedded instructions
        constraints = self.parse_constraints(raw_input)
        preferences = self.parse_preferences(raw_input)
        success_criteria = self.parse_success_criteria(raw_input)
        
        return EnhancedInput(
            original_message=raw_input,
            intent=intent,
            priority=priority,
            context=context,
            constraints=constraints,
            preferences=preferences,
            success_criteria=success_criteria,
            enhanced_prompt=self.build_enhanced_prompt(...)
        )
```

### **2. Context Enrichment**
```python
# User says: "Add authentication to the API"
# System enriches to:
enhanced_input = {
    "original": "Add authentication to the API",
    "intent": "implement_feature",
    "domain": "security",
    "complexity": "medium",
    "context": {
        "project_type": "web_api",
        "existing_files": ["app.py", "models.py", "routes.py"],
        "dependencies": ["flask", "sqlalchemy"],
        "security_level": "standard"
    },
    "implied_tasks": [
        "Create user model",
        "Add authentication middleware", 
        "Implement login/logout endpoints",
        "Add password hashing",
        "Create authentication tests"
    ],
    "success_criteria": [
        "Users can register and login",
        "Protected endpoints require authentication",
        "Passwords are securely hashed",
        "Tests pass"
    ]
}
```

### **3. Agent-Specific Instruction Generation**
```python
def generate_agent_instructions(enhanced_input: EnhancedInput, agent_type: AgentType) -> str:
    """Create agent-specific instructions with embedded expectations."""
    
    base_template = """
    You are {agent_name} working in an orchestrated environment.
    
    TASK CONTEXT:
    - Intent: {intent}
    - Complexity: {complexity} 
    - Priority: {priority}
    - Success Criteria: {criteria}
    
    COMMUNICATION REQUIREMENTS:
    Use these XML tags for progress updates:
    
    <STATUS>{current_status}</STATUS> - Update as you work
    <CURRENT_ACTIVITY>What you're doing now</CURRENT_ACTIVITY>
    <PROGRESS>step: X/Y or percentage</PROGRESS>
    <CONFIDENCE>high|medium|low based on task clarity</CONFIDENCE>
    
    For each major action, use:
    <ACTIONS_COMPLETED>- Action description</ACTIONS_COMPLETED>
    <FILES_CREATED>/path/to/file</FILES_CREATED>
    <FILES_MODIFIED>/path/to/file - what changed</FILES_MODIFIED>
    
    If you need clarification:
    <QUESTIONS>
      <CLARIFICATION>Specific question about requirements</CLARIFICATION>
      <PERMISSION>Request to do something potentially destructive</PERMISSION>
      <TECHNICAL>Question about implementation approach</TECHNICAL>
    </QUESTIONS>
    
    At completion:
    <SUMMARY>Brief description of what was accomplished</SUMMARY>
    <TEST_RESULTS>Any test outcomes</TEST_RESULTS>
    <RECOMMENDATIONS>Suggestions for follow-up work</RECOMMENDATIONS>
    
    USER REQUEST: {original_message}
    """
    
    return base_template.format(
        agent_name=get_agent_name(agent_type),
        intent=enhanced_input.intent,
        complexity=enhanced_input.complexity,
        priority=enhanced_input.priority,
        criteria=enhanced_input.success_criteria,
        original_message=enhanced_input.original_message
    )
```

---

## 🎨 **Output Languagization System**

### **1. Progressive Status Interpretation**
```python
class StatusLanguagizer:
    def interpret_status_progression(self, status_history: List[StatusUpdate]) -> str:
        """Convert status updates into natural language narrative."""
        
        narratives = []
        
        for i, update in enumerate(status_history):
            if i == 0:
                narratives.append(f"🚀 Started by {self.action_to_language(update.activity)}")
            
            elif update.status == "blocked":
                narratives.append(f"⏸️ Hit a roadblock: {update.current_activity}")
                if update.questions:
                    narratives.append(f"   Needs clarification on: {update.questions[0]}")
            
            elif update.status == "working":
                progress_phrase = self.progress_to_language(update.progress)
                narratives.append(f"⚡ Making progress: {update.current_activity} ({progress_phrase})")
            
            elif update.status == "completed":
                summary = self.summarize_accomplishments(update)
                narratives.append(f"✅ Finished successfully: {summary}")
        
        return "\n".join(narratives)
    
    def progress_to_language(self, progress: str) -> str:
        """Convert progress indicators to natural language."""
        if "%" in progress:
            pct = int(progress.replace("%", ""))
            if pct < 25: return "just getting started"
            elif pct < 50: return "making good headway"
            elif pct < 75: return "well underway"
            elif pct < 95: return "nearly there"
            else: return "finishing up"
        
        elif "step:" in progress:
            current, total = progress.replace("step: ", "").split("/")
            return f"step {current} of {total}"
        
        return progress
```

### **2. Action Summarization**
```python
class ActionLanguagizer:
    def summarize_actions(self, actions: List[str]) -> str:
        """Convert action lists into coherent narrative."""
        
        if not actions:
            return "No major actions taken."
        
        # Categorize actions
        file_actions = [a for a in actions if any(word in a.lower() for word in ['created', 'modified', 'deleted'])]
        code_actions = [a for a in actions if any(word in a.lower() for word in ['implemented', 'added', 'fixed'])]
        test_actions = [a for a in actions if 'test' in a.lower()]
        config_actions = [a for a in actions if any(word in a.lower() for word in ['configured', 'setup', 'installed'])]
        
        summary_parts = []
        
        if file_actions:
            file_count = len([a for a in file_actions if 'created' in a.lower()])
            if file_count > 0:
                summary_parts.append(f"Created {file_count} new files")
        
        if code_actions:
            summary_parts.append(f"Implemented {len(code_actions)} code changes")
        
        if test_actions:
            summary_parts.append(f"Added {len(test_actions)} tests")
        
        if config_actions:
            summary_parts.append(f"Configured {len(config_actions)} components")
        
        # Create natural language summary
        if len(summary_parts) == 1:
            return summary_parts[0]
        elif len(summary_parts) == 2:
            return f"{summary_parts[0]} and {summary_parts[1]}"
        else:
            return f"{', '.join(summary_parts[:-1])}, and {summary_parts[-1]}"
```

### **3. Intelligent Question Formatting**
```python
class QuestionLanguagizer:
    def format_questions(self, questions: Dict[str, List[str]]) -> str:
        """Format questions by category with appropriate urgency."""
        
        formatted = []
        
        # High priority questions first
        if questions.get("PERMISSION"):
            formatted.append("🚨 **Requires Permission:**")
            for q in questions["PERMISSION"]:
                formatted.append(f"   • {q}")
        
        if questions.get("CLARIFICATION"):
            formatted.append("❓ **Needs Clarification:**") 
            for q in questions["CLARIFICATION"]:
                formatted.append(f"   • {q}")
        
        if questions.get("TECHNICAL"):
            formatted.append("🔧 **Technical Decisions:**")
            for q in questions["TECHNICAL"]:
                formatted.append(f"   • {q}")
        
        if questions.get("PREFERENCE"):
            formatted.append("⚙️ **Preferences:**")
            for q in questions["PREFERENCE"]:
                formatted.append(f"   • {q}")
        
        return "\n".join(formatted)
```

---

## 🔗 **Bidirectional Communication Protocol**

### **1. Lead Agent → Sub-Agent Communication**
```python
class LeadAgentCommunicator:
    def send_instructions(self, agent_id: str, enhanced_input: EnhancedInput) -> TaskSession:
        """Send enhanced instructions to sub-agent."""
        
        instruction_packet = {
            "task_id": generate_task_id(),
            "agent_id": agent_id,
            "priority": enhanced_input.priority,
            "context": enhanced_input.context,
            "instructions": enhanced_input.enhanced_prompt,
            "success_criteria": enhanced_input.success_criteria,
            "communication_requirements": {
                "status_updates": "required",
                "progress_frequency": "every_major_step", 
                "question_escalation": "immediate",
                "completion_summary": "detailed"
            },
            "expected_tags": ["STATUS", "PROGRESS", "CURRENT_ACTIVITY", "ACTIONS_COMPLETED", "FILES_CREATED"]
        }
        
        return self.dispatch_to_agent(instruction_packet)
    
    def process_agent_response(self, response: AgentResponse) -> LeadAgentDecision:
        """Process structured response and decide next steps."""
        
        if response.status == "needs_input":
            return self.handle_questions(response.questions)
        elif response.status == "blocked":
            return self.resolve_blocking_issues(response)
        elif response.status == "completed":
            return self.validate_completion(response)
        elif response.status in ["working", "analyzing"]:
            return self.monitor_progress(response)
        
        return LeadAgentDecision.CONTINUE_MONITORING
```

### **2. Sub-Agent → Lead Agent Communication**
```python
class SubAgentCommunicator:
    def send_status_update(self, status: str, activity: str, progress: str = None):
        """Send real-time status update."""
        
        update = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
            "current_activity": activity,
            "progress": progress,
            "session_id": self.session_id
        }
        
        self.broadcast_update(update)
    
    def request_clarification(self, question_type: str, question: str, urgency: str = "normal"):
        """Request clarification from lead agent."""
        
        clarification_request = {
            "type": "question",
            "category": question_type,
            "question": question,
            "urgency": urgency,
            "context": self.current_context,
            "blocking": question_type == "PERMISSION"
        }
        
        return self.send_to_lead(clarification_request)
```

---

## 🎯 **Integration & Intuitive UX Design**

### **1. Real-Time Progress Visualization**
```
Lead Agent Dashboard:

┌─ Task: Implement User Authentication ─────────────────────┐
│ Agent: Claude Code                    Status: Working ⚡  │
│ Progress: ████████░░ 75% (Step 6/8)                      │
│ Activity: Running integration tests                       │
│ ETA: ~3 minutes                                          │
├───────────────────────────────────────────────────────────┤
│ Recent Actions:                                           │
│ ✓ Created user model and authentication middleware       │
│ ✓ Added password hashing with bcrypt                     │  
│ ✓ Implemented login/logout endpoints                     │
│ ⚡ Running test suite (24/30 tests passing)              │
├───────────────────────────────────────────────────────────┤
│ Files: 📄 3 created, 📝 2 modified                       │
│ Questions: None pending                                   │
└───────────────────────────────────────────────────────────┘
```

### **2. Conversational Status Updates**
```
Natural Language Progress Feed:

🚀 Claude started by analyzing the existing codebase structure
⚡ Making good progress: implementing user authentication (step 3 of 7) 
📄 Created new files: user_model.py, auth_middleware.py
⚡ Well underway: adding password hashing and session management (65%)
🧪 Running tests: 24 passed, 2 failed, coverage at 87%
❓ Quick question: Should I add 2FA support or keep it simple for now?
⚡ Continuing with current approach based on your "keep simple" preference
✅ Finished successfully: Complete authentication system with tests passing
```

### **3. Smart Intervention Points**
```python
class InterventionManager:
    def should_interrupt(self, status_update: StatusUpdate) -> bool:
        """Determine if lead agent should intervene."""
        
        # Always interrupt for questions requiring permission
        if any(q.category == "PERMISSION" for q in status_update.questions):
            return True
            
        # Interrupt if blocked for more than threshold
        if status_update.status == "blocked" and status_update.blocked_duration > timedelta(minutes=2):
            return True
            
        # Interrupt if confidence drops significantly
        if status_update.confidence == "low" and self.previous_confidence != "low":
            return True
            
        # Interrupt if approaching resource limits
        if status_update.resources.tokens_consumed > 0.8 * self.token_limit:
            return True
            
        return False
```

### **4. Context-Aware Response Generation**
```python
class ContextAwareResponder:
    def generate_response(self, question: Question, context: TaskContext) -> str:
        """Generate contextually appropriate responses."""
        
        if question.category == "CLARIFICATION":
            # Provide clarification based on task context and history
            return self.clarify_based_on_context(question, context)
            
        elif question.category == "PERMISSION":
            # Auto-approve safe operations, escalate risky ones
            risk_level = self.assess_risk(question, context)
            if risk_level == "low":
                return "Approved - proceed with the change"
            else:
                return f"Please confirm: {question.question} (Risk: {risk_level})"
                
        elif question.category == "TECHNICAL":
            # Provide technical guidance based on project standards
            return self.provide_technical_guidance(question, context)
```

---

## 📋 **Implementation Phases**

### **Phase 1: Core Tag System** ✅
- [x] Basic XML tags (STATUS, ACTIONS, FILES, QUESTIONS)
- [x] Parsing and formatting system  
- [x] Agent prompt injection

### **Phase 2: Advanced Tags** 🚧
- [ ] Progress tracking and confidence levels
- [ ] Resource usage monitoring
- [ ] Test results and quality metrics
- [ ] Temporal status updates

### **Phase 3: Input Transformation** 📋
- [ ] Intent recognition and preprocessing
- [ ] Context enrichment pipeline
- [ ] Agent-specific instruction generation
- [ ] Success criteria extraction

### **Phase 4: Output Languagization** 📋
- [ ] Progressive status interpretation
- [ ] Action summarization
- [ ] Question formatting and prioritization
- [ ] Natural language progress feeds

### **Phase 5: Bidirectional Communication** 📋
- [ ] Real-time status streaming
- [ ] Question escalation system
- [ ] Context-aware response generation
- [ ] Smart intervention points

### **Phase 6: Integration & UX** 📋
- [ ] Progress visualization dashboard
- [ ] Conversational status updates
- [ ] Context-aware automation
- [ ] Performance optimization

---

## 🎯 **Benefits & Outcomes**

### **For Lead Agent Orchestration:**
1. **Real-time Understanding**: Know exactly what sub-agents are doing
2. **Intelligent Intervention**: Auto-detect when guidance is needed
3. **Context Preservation**: Maintain rich context across agent interactions
4. **Quality Assurance**: Track test results, code quality, and validation
5. **Resource Management**: Monitor token usage, time, and computational resources

### **For Sub-Agent Performance:**
1. **Clear Expectations**: Understand exactly how to communicate progress
2. **Structured Feedback**: Provide actionable information to the orchestrator
3. **Escalation Paths**: Clear channels for questions and clarification
4. **Quality Metrics**: Self-assess and report on work quality
5. **Context Awareness**: Understand the broader task context

### **For User Experience:**
1. **Transparency**: See exactly what agents are doing and why
2. **Control**: Clear intervention points for guidance and approval
3. **Efficiency**: Reduced back-and-forth through structured communication
4. **Reliability**: Consistent, parseable agent outputs
5. **Insight**: Rich metrics on agent performance and task progress

This comprehensive communication protocol transforms agent orchestration from reactive monitoring to proactive, intelligent coordination with rich, structured communication that's both machine-parseable and human-intuitive.
