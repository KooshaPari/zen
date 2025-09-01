Kafka Setup (Local Dev)

This repo includes a single-broker Kafka stack for local development using Docker Compose.

What’s included
- Zookeeper (Confluent image)
- Kafka broker (Confluent image)
- Kafdrop UI at http://localhost:19000

Start services
- docker compose up -d kafka zookeeper kafdrop

Configure the server
- If running Zen MCP server on your host: export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
- If running Zen MCP in the compose stack: it already uses kafka:29092 via the compose network.

Verify connectivity
- UI: open http://localhost:19000 and confirm you can see the broker
- Auto-create topics is enabled. When the server starts, it sends a message to a connection-test topic; topics used by the app will be created on first publish.
- App health: once the server is running, hit http://localhost:<port>/health/kafka to verify connection and a test publish.

Common issues
- Connection refused: ensure ports 9092 (host) and 29092 (container) are not blocked or used by another process.
- Broker advertised listeners: compose uses advertised listeners for both internal (kafka:29092) and external (localhost:9092) access.
- IPv6: if your host prefers IPv6 and you see IPv6 connection errors, forcing IPv4 with KAFKA_ADVERTISED_LISTENERS using localhost should resolve it (as configured).

Cleanup
- docker compose down
- docker compose down -v  # also remove volumes if you want a clean slate
