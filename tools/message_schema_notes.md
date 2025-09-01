Message schema rollout notes:

- Introduced tools.shared.agent_models.Message with fields:
  - role: system|user|assistant|tool (string)
  - content: Any (may be string or structured provider-specific objects)
  - message: string (legacy mirror of content)
  - time: string (ISO8601)
- AgentTaskResult.messages now typed as list[Message]
- For compatibility, existing producers that build dicts will still validate because Pydantic accepts dict -> Message
- Consumers should migrate to prefer `content` over `message`

