# CRUD Todo App with Agent Orchestration

This example demonstrates how to build a complete CRUD todo application using zen-mcp-server's agent orchestration capabilities. We'll decompose the project into specialized tasks and assign them to the most appropriate agents.

## Project Overview

**Goal**: Build a full-stack todo application with:
- React frontend with TypeScript
- Node.js/Express backend with PostgreSQL
- User authentication
- CRUD operations for todos
- Modern UI/UX design
- Comprehensive testing
- Performance optimization

## Step-by-Step Implementation

### Phase 1: Project Planning and Setup

#### 1.1 Discover Available Agents
```json
{
  "tool": "agent_registry",
  "arguments": {
    "check_availability": true,
    "include_capabilities": true
  }
}
```

#### 1.2 Project Structure Setup
```json
{
  "tool": "agent_sync",
  "arguments": {
    "agent_type": "goose",
    "task_description": "Project initialization and structure setup",
    "message": "Create a full-stack todo app project structure:\n\n1. Initialize root directory with:\n   - frontend/ (React + TypeScript)\n   - backend/ (Node.js + Express)\n   - database/ (PostgreSQL migrations)\n   - docs/ (API documentation)\n   - docker-compose.yml for development\n\n2. Set up package.json files with appropriate dependencies\n3. Create basic configuration files (.env.example, .gitignore, etc.)\n4. Initialize git repository with proper .gitignore",
    "working_directory": "./todo-app",
    "timeout_seconds": 300
  }
}
```

### Phase 2: Parallel Development - Core Components

#### 2.1 Launch Parallel Development Tasks
```json
{
  "tool": "agent_batch",
  "arguments": {
    "batch_description": "Core Todo App Development - Phase 1",
    "coordination_strategy": "parallel",
    "max_concurrent": 4,
    "fail_fast": false,
    "timeout_seconds": 2400,
    "tasks": [
      {
        "agent_type": "claude",
        "task_description": "Frontend: Core Components and Types",
        "message": "Create the core frontend components for the todo app:\n\n1. TypeScript types and interfaces:\n   - User, Todo, AuthState, ApiResponse types\n   - Proper enum definitions for todo status\n\n2. Core React components:\n   - TodoItem component with edit/delete/toggle functionality\n   - TodoList component with filtering and sorting\n   - AddTodo component with form validation\n   - Header component with user info and logout\n\n3. Custom hooks:\n   - useTodos for todo state management\n   - useAuth for authentication state\n   - useApi for API calls with error handling\n\n4. Utility functions:\n   - API client with proper error handling\n   - Date formatting helpers\n   - Validation utilities\n\nUse modern React patterns (hooks, context) and ensure full TypeScript coverage.",
        "working_directory": "./todo-app/frontend",
        "agent_args": ["--allowedTools", "Edit Replace"],
        "priority": "high"
      },
      {
        "agent_type": "aider",
        "task_description": "Backend: Database Schema and Models",
        "message": "Create the backend database layer:\n\n1. PostgreSQL database schema:\n   - users table (id, email, password_hash, created_at, updated_at)\n   - todos table (id, user_id, title, description, completed, priority, due_date, created_at, updated_at)\n   - Proper foreign key relationships and indexes\n\n2. Database migrations:\n   - Initial schema creation\n   - Seed data for development\n\n3. Sequelize/Prisma models:\n   - User model with authentication methods\n   - Todo model with validation\n   - Proper associations between models\n\n4. Database connection and configuration:\n   - Environment-based configuration\n   - Connection pooling\n   - Error handling\n\nEnsure proper data validation and security best practices.",
        "working_directory": "./todo-app/backend",
        "priority": "high"
      },
      {
        "agent_type": "claude",
        "task_description": "Backend: Authentication System",
        "message": "Implement a secure authentication system:\n\n1. JWT-based authentication:\n   - User registration with email validation\n   - User login with password hashing (bcrypt)\n   - JWT token generation and validation\n   - Refresh token mechanism\n\n2. Authentication middleware:\n   - Protect routes requiring authentication\n   - Extract user info from JWT tokens\n   - Handle token expiration gracefully\n\n3. Password security:\n   - Strong password requirements\n   - Secure password hashing\n   - Password reset functionality (email-based)\n\n4. API endpoints:\n   - POST /auth/register\n   - POST /auth/login\n   - POST /auth/refresh\n   - POST /auth/logout\n   - POST /auth/forgot-password\n   - POST /auth/reset-password\n\nInclude proper error handling, validation, and security headers.",
        "working_directory": "./todo-app/backend",
        "priority": "high"
      },
      {
        "agent_type": "aider",
        "task_description": "Backend: Todo CRUD API",
        "message": "Implement comprehensive Todo CRUD API:\n\n1. Todo API endpoints:\n   - GET /api/todos (with filtering, sorting, pagination)\n   - POST /api/todos (create new todo)\n   - GET /api/todos/:id (get specific todo)\n   - PUT /api/todos/:id (update todo)\n   - DELETE /api/todos/:id (delete todo)\n   - PATCH /api/todos/:id/toggle (toggle completion status)\n\n2. Advanced features:\n   - Filtering by status, priority, due date\n   - Sorting by creation date, due date, priority\n   - Pagination with limit/offset\n   - Search functionality in title/description\n\n3. Validation and error handling:\n   - Input validation using Joi or similar\n   - Proper HTTP status codes\n   - Detailed error messages\n   - Rate limiting\n\n4. Authorization:\n   - Ensure users can only access their own todos\n   - Proper permission checks\n\nInclude comprehensive API documentation with examples.",
        "working_directory": "./todo-app/backend",
        "priority": "normal"
      }
    ]
  }
}
```

### Phase 3: Monitor Progress and Integration

#### 3.1 Check Development Progress
```json
{
  "tool": "agent_inbox",
  "arguments": {
    "action": "list",
    "filter_status": ["running", "completed"]
  }
}
```

#### 3.2 Get Detailed Results
```json
{
  "tool": "agent_inbox",
  "arguments": {
    "task_id": "frontend-core-task-id",
    "action": "results",
    "include_messages": true,
    "max_message_length": 2000
  }
}
```

### Phase 4: UI/UX and Advanced Features

#### 4.1 Launch UI/UX Enhancement Tasks
```json
{
  "tool": "agent_batch",
  "arguments": {
    "batch_description": "Todo App UI/UX and Advanced Features",
    "coordination_strategy": "parallel",
    "max_concurrent": 3,
    "tasks": [
      {
        "agent_type": "claude",
        "task_description": "Frontend: Modern UI Design",
        "message": "Create a modern, responsive UI design:\n\n1. Design system:\n   - Color palette and typography\n   - Component library with consistent styling\n   - Responsive breakpoints\n   - Dark/light theme support\n\n2. Advanced UI components:\n   - Drag-and-drop todo reordering\n   - Animated transitions and micro-interactions\n   - Loading states and skeletons\n   - Toast notifications for actions\n   - Modal dialogs for confirmations\n\n3. Accessibility:\n   - ARIA labels and roles\n   - Keyboard navigation\n   - Screen reader support\n   - High contrast mode\n\n4. Mobile optimization:\n   - Touch-friendly interactions\n   - Swipe gestures for actions\n   - Responsive layout\n\nUse CSS-in-JS (styled-components) or Tailwind CSS for styling.",
        "working_directory": "./todo-app/frontend",
        "priority": "normal"
      },
      {
        "agent_type": "claude",
        "task_description": "Frontend: State Management and Performance",
        "message": "Implement advanced state management and performance optimizations:\n\n1. State management:\n   - Context API or Redux Toolkit for global state\n   - Optimistic updates for better UX\n   - Offline support with local storage sync\n   - Real-time updates (WebSocket integration)\n\n2. Performance optimizations:\n   - React.memo for component optimization\n   - useMemo and useCallback for expensive operations\n   - Virtual scrolling for large todo lists\n   - Image optimization and lazy loading\n   - Code splitting and lazy loading of routes\n\n3. Error boundaries and error handling:\n   - Global error boundary\n   - Retry mechanisms for failed API calls\n   - User-friendly error messages\n   - Error reporting integration\n\n4. Testing setup:\n   - Jest and React Testing Library configuration\n   - Unit tests for components and hooks\n   - Integration tests for user flows\n   - E2E test setup with Cypress",
        "working_directory": "./todo-app/frontend",
        "priority": "normal"
      },
      {
        "agent_type": "goose",
        "task_description": "DevOps: Docker and Deployment Setup",
        "message": "Set up containerization and deployment infrastructure:\n\n1. Docker configuration:\n   - Multi-stage Dockerfile for frontend (build + serve)\n   - Dockerfile for backend with proper optimization\n   - Docker Compose for development environment\n   - PostgreSQL container configuration\n\n2. Development workflow:\n   - Hot reloading for both frontend and backend\n   - Database seeding and migration scripts\n   - Environment variable management\n   - Health checks for all services\n\n3. Production deployment:\n   - Production-optimized Docker images\n   - Nginx configuration for frontend serving\n   - SSL/TLS certificate setup\n   - Environment-specific configurations\n\n4. CI/CD pipeline setup:\n   - GitHub Actions or similar for automated testing\n   - Automated deployment to staging/production\n   - Database migration automation\n   - Security scanning and dependency updates",
        "working_directory": "./todo-app",
        "priority": "low"
      }
    ]
  }
}
```

### Phase 5: Testing and Quality Assurance

#### 5.1 Comprehensive Testing
```json
{
  "tool": "agent_async",
  "arguments": {
    "agent_type": "claude",
    "task_description": "Comprehensive Testing Suite",
    "message": "Create a comprehensive testing suite for the todo app:\n\n1. Backend testing:\n   - Unit tests for all API endpoints\n   - Integration tests for database operations\n   - Authentication and authorization tests\n   - Performance tests for API endpoints\n   - Security tests for common vulnerabilities\n\n2. Frontend testing:\n   - Component unit tests with React Testing Library\n   - Hook testing for custom hooks\n   - Integration tests for user workflows\n   - Accessibility testing with axe-core\n   - Visual regression tests\n\n3. End-to-end testing:\n   - Complete user journey tests\n   - Cross-browser compatibility tests\n   - Mobile responsiveness tests\n   - Performance testing with Lighthouse\n\n4. Test automation:\n   - Automated test runs on CI/CD\n   - Code coverage reporting\n   - Test result notifications\n   - Performance monitoring\n\nAim for >90% code coverage and comprehensive user scenario coverage.",
    "timeout_seconds": 1800,
    "priority": "high"
  }
}
```

### Phase 6: Final Integration and Polish

#### 6.1 Integration and Bug Fixes
```json
{
  "tool": "agent_sync",
  "arguments": {
    "agent_type": "aider",
    "task_description": "Final Integration and Bug Fixes",
    "message": "Perform final integration and resolve any issues:\n\n1. Integration testing:\n   - Test frontend-backend communication\n   - Verify all API endpoints work correctly\n   - Test authentication flow end-to-end\n   - Validate data persistence and retrieval\n\n2. Bug fixes and optimizations:\n   - Fix any integration issues\n   - Optimize database queries\n   - Improve error handling\n   - Polish user experience\n\n3. Documentation:\n   - API documentation with examples\n   - Frontend component documentation\n   - Deployment guide\n   - User manual\n\n4. Final polish:\n   - Code cleanup and refactoring\n   - Performance optimizations\n   - Security review\n   - Accessibility audit",
    "timeout_seconds": 900
  }
}
```

## Monitoring and Results

### Check Final Status
```json
{
  "tool": "agent_inbox",
  "arguments": {
    "action": "list"
  }
}
```

### Get Complete Project Results
```json
{
  "tool": "agent_inbox",
  "arguments": {
    "task_id": "integration-task-id",
    "action": "results",
    "include_messages": true
  }
}
```

## Expected Outcomes

After completing this orchestrated development process, you should have:

1. **Complete Full-Stack Application**
   - Modern React frontend with TypeScript
   - Robust Node.js backend with PostgreSQL
   - Secure authentication system
   - Full CRUD functionality for todos

2. **Production-Ready Features**
   - Responsive design with dark/light themes
   - Offline support and real-time updates
   - Comprehensive error handling
   - Performance optimizations

3. **Quality Assurance**
   - >90% test coverage
   - Accessibility compliance
   - Security best practices
   - Performance optimization

4. **DevOps Infrastructure**
   - Containerized deployment
   - CI/CD pipeline
   - Monitoring and logging
   - Automated testing

## Benefits of Agent Orchestration

This approach demonstrates the power of agent orchestration:

- **Parallel Development**: Multiple aspects developed simultaneously
- **Specialized Expertise**: Each agent handles tasks suited to their strengths
- **Faster Delivery**: Reduced overall development time
- **Higher Quality**: Specialized focus leads to better outcomes
- **Scalability**: Easy to add more agents or tasks as needed

The entire project can be completed in a fraction of the time it would take with sequential development, while maintaining high quality and comprehensive coverage of all aspects.
