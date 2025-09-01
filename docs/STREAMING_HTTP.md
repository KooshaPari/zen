Streamable HTTP API (Real-time Streaming)

Endpoints

- POST /tasks: Create and start a new task (202 Accepted)
- GET /tasks/{id}: Snapshot of the task (JSON)
- GET /tasks/{id}/results: Final result when ready
- POST /tasks/{id}/cancel: Best-effort cancel
- GET /tasks/{id}/events: Real-time SSE (status/progress/action/file/heartbeat/completion)
- GET /events/live: Global real-time SSE for all tasks
- GET /tasks: List tasks (filters: agent, status, dates)
- GET /tasks.csv: CSV export
- GET /health, GET /readyz

Quick start

1) Start the server:
   python -m server_http --host 0.0.0.0 --port 8080

2) Create a task:
   curl -sS -X POST http://localhost:8080/tasks \
     -H 'Content-Type: application/json' \
     -d '{
           "agent_type":"claude",
           "task_description":"Fix lints",
           "message":"Please fix ruff errors in utils/"
         }'

   Response (202): {"task_id":"<id>","status":"pending"}

3) Stream real-time events for that task:
   curl -N http://localhost:8080/tasks/<id>/events

4) Or watch a global stream (all tasks):
   curl -N http://localhost:8080/events/live

5) Cancel a task:
   curl -sS -X POST http://localhost:8080/tasks/<id>/cancel

Notes

- /tasks/{id}/events is the recommended live stream. Polling SSE at /tasks/{id}/stream remains for compatibility.
- The fastapply_edit tool can include a task_id to emit file-update events into the live streams.
- For idempotency, clients may use an Idempotency-Key header on POST /tasks.
- All SSE endpoints use Content-Type: text/event-stream and support long-lived connections. Consider setting X-Accel-Buffering: no when proxying through Nginx.
- GET /tasks.csv returns an ETag and supports If-None-Match for efficient caching.
