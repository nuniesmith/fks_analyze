# Expanded Standard AI Chat Prompts for Codebase Management

This is an expanded version of the previous Markdown document. I've reviewed the provided repo list again (noting the mix of private and public repos, languages like Python, Rust, C#, TypeScript, Shell, HTML, Jupyter Notebook, Java, and Batchfile; focus on MIT licenses; recent updates; and themes around "fks_" for core services, "shared_" for DRY utilities, "personal_games_" for games, and "personal_services/servers_" for personal tools). The expansion includes:

- New categories: Testing & Quality Assurance, Security & Compliance, Scaling & Deployment, Collaboration & Version Control.
- Additional prompts in existing categories for better coverage (e.g., more on Kubernetes prep, API integrations).
- Emphasis on handling up to 32 repos, promoting DRY by prioritizing shared repos (e.g., `shared_docker`, `shared_scripts`, `shared_schema`), and easing new project rollouts.
- Prompts now include more placeholders for specificity and reminders to reference the full repo ecosystem.

As before, use these as templates—fill in placeholders and prepend with context if needed.

## General Guidelines for Using These Prompts
- **Context Inclusion**: Prepend with "My repos: [brief list or summary, e.g., fks_api (Python service), shared_docker (Shell for containers), personal_games_clonehero (Python game), etc. Up to 32 repos total."
- **DRY Focus**: Always encourage moving common code/configs to shared repos.
- **Tech Stack Reminder**: "Stack: Docker/Docker Compose (K8s future); langs: Python, Rust, C#, Node.js/JS/TS/React, HTML/CSS, shell scripts, plus Jupyter, Java, Batchfile in some repos."
- **Scalability**: For larger tasks, mention "considering scale across 32+ repos."
- **Update Handling**: If repos change, note recent update times (e.g., many updated 5 hours ago).

## 1. Code Review and Refactoring Prompts
(Expanded with prompts for API-specific reviews and dependency management.)

- **General Code Review**:
  ```
  Review the following code from [REPO_NAME] ([LANGUAGE]): [PASTE_CODE_HERE]. Suggest improvements for performance, readability, and DRY principles. If applicable, recommend moving shared logic to a shared repo like shared_[LANGUAGE] or shared_scripts. Consider our stack: Docker/Docker Compose, etc.
  ```

- **DRY-Focused Refactoring**:
  ```
  Analyze this code snippet from [REPO_NAME] ([LANGUAGE]): [PASTE_CODE_HERE]. Identify duplicated patterns and suggest refactoring to make it DRY. Propose extracting common parts to a shared repo (e.g., shared_python for Python utils, shared_rust for Rust crates). Ensure compatibility with Docker setups in shared_docker.
  ```

- **Cross-Repo Consistency Check**:
  ```
  Compare code patterns across these repos: [LIST_REPOS, e.g., fks_api, fks_engine, shared_python]. Look for inconsistencies in [SPECIFIC_ASPECT, e.g., error handling, logging]. Suggest standardizations to promote DRY, using shared repos where possible.
  ```

- **API Endpoint Review (e.g., for fks_api, fks_auth)**:
  ```
  Review this API code from [REPO_NAME] ([LANGUAGE]): [PASTE_CODE_HERE]. Check for REST/GraphQL best practices, security (e.g., auth), and performance. Suggest DRY integrations with shared_schema for data models or shared_python for utils.
  ```

- **Dependency Audit**:
  ```
  Audit dependencies in [REPO_NAME] ([LANGUAGE]). List outdated/vulnerable packages and suggest updates. Propose centralizing common deps in a shared repo like shared_python or shared_rust for DRY management across 32 repos.
  ```

## 2. Docker and Containerization Prompts
(Expanded with multi-repo orchestration and K8s prep prompts.)

- **Dockerfile Optimization**:
  ```
  Optimize this Dockerfile from [REPO_NAME]: [PASTE_DOCKERFILE_HERE]. Focus on multi-stage builds, security, and efficiency for [LANGUAGE]. Integrate best practices from shared_docker if relevant. Prepare for future K8s migration.
  ```

- **Docker Compose Setup**:
  ```
  Help set up a docker-compose.yml for [REPO_NAME] integrating services from [RELATED_REPOS, e.g., fks_nginx, fks_api]. Include volumes, networks, and env vars. Draw from shared_docker and shared_scripts for reusable configs. Ensure scalability for K8s.
  ```

- **Migration to Kubernetes**:
  ```
  Outline steps to migrate this Docker Compose setup from [REPO_NAME] to Kubernetes: [PASTE_COMPOSE_YML_HERE]. Suggest Helm charts or manifests, focusing on shared resources from shared_docker or shared_nginx.
  ```

- **Multi-Repo Orchestration**:
  ```
  Create a top-level Docker Compose or K8s config to orchestrate [LIST_REPOS, e.g., fks_api, fks_worker, fks_nginx]. Use shared_docker for base images and shared_scripts for entrypoints. Focus on DRY for easy rollout of new services.
  ```

- **Container Security Scan**:
  ```
  Suggest tools and scripts to scan Docker images in [REPO_NAME]. Integrate with shared_actions for CI. Prepare for K8s security policies.
  ```

## 3. Language-Specific Development Prompts
(Expanded with Jupyter-specific and mixed-language integration prompts.)

- **Python-Specific (e.g., for fks_api, fks_data)**:
  ```
  Write or refactor Python code for [TASK_DESCRIPTION] in [REPO_NAME]. Use best practices like type hints and async if needed. Check for DRY opportunities by integrating from shared_python. Test with Docker via shared_docker.
  ```

- **Rust-Specific (e.g., for fks_analyze, shared_rust)**:
  ```
  Implement [FEATURE_DESCRIPTION] in Rust for [REPO_NAME]. Emphasize safety, performance, and error handling. Suggest crates from shared_rust to avoid duplication. Build and test in Docker using shared_scripts.
  ```

- **C#-Specific (e.g., for fks_ninja)**:
  ```
  Develop C# code for [TASK_DESCRIPTION] in [REPO_NAME]. Focus on .NET best practices and async patterns. Extract shared utilities to a new or existing shared repo if DRY violations are found.
  ```

- **Node.js/JS/TS/React-Specific (e.g., for fks_web, shared_react)**:
  ```
  Create or improve [COMPONENT/PAGE_DESCRIPTION] in [JS/TS/React] for [REPO_NAME]. Use modern hooks and state management. Promote DRY by reusing components from shared_react. Bundle with Docker and nginx from shared_nginx.
  ```

- **HTML/CSS-Specific (e.g., for shared_nginx, personal_services_dashboard)**:
  ```
  Design or refine HTML/CSS for [ELEMENT_DESCRIPTION] in [REPO_NAME]. Ensure responsive design and accessibility. Share common styles in a shared repo if applicable.
  ```

- **Shell Script-Specific (e.g., for shared_scripts, fks_config)**:
  ```
  Write a shell script for [TASK_DESCRIPTION, e.g., deployment automation] in [REPO_NAME]. Make it idempotent and portable. Integrate with shared_scripts for common functions and Docker commands from shared_docker.
  ```

- **Jupyter Notebook-Specific (e.g., for fks_master)**:
  ```
  Create or optimize a Jupyter Notebook for [TASK_DESCRIPTION, e.g., data analysis] in [REPO_NAME]. Include visualizations and Markdown docs. Suggest DRY by extracting reusable cells to shared_python.
  ```

- **Java-Specific (e.g., for personal_games_2009scape)**:
  ```
  Implement [FEATURE_DESCRIPTION] in Java for [REPO_NAME]. Focus on OOP best practices. Integrate shared utils if applicable, and containerize with shared_docker.
  ```

- **Batchfile-Specific (e.g., for personal_games_ats)**:
  ```
  Write a Batchfile script for [TASK_DESCRIPTION] in [REPO_NAME]. Ensure cross-platform compatibility where possible. Link to shared_scripts for advanced automation.
  ```

- **Mixed-Language Integration**:
  ```
  Suggest how to integrate code from [REPO1_LANGUAGE] in [REPO1_NAME] with [REPO2_LANGUAGE] in [REPO2_NAME], e.g., Rust FFI with Python. Use Docker for runtime and shared repos for interfaces.
  ```

## 4. Repo Management and New Project Rollout Prompts
(Expanded with repo merging and template creation prompts.)

- **New Repo Setup**:
  ```
  Guide me on setting up a new repo named [NEW_REPO_NAME] for [PURPOSE, e.g., a Python service]. Include initial structure, LICENSE (MIT), and integrations with shared repos (e.g., shared_docker for containers, shared_scripts for CI). Focus on DRY to make rollout easy across 32+ repos.
  ```

- **Shared Repo Integration**:
  ```
  Suggest how to integrate [SHARED_REPO, e.g., shared_python] into [TARGET_REPO, e.g., fks_worker]. Provide code examples for importing/utils. Ensure DRY across my 32 repos.
  ```

- **Repo Consolidation for DRY**:
  ```
  Review these repos: [LIST_REPOS, e.g., fks_auth, fks_data, personal_games_coc]. Identify overlapping code/features and propose moving them to a shared repo (e.g., create/update shared_schema). Provide migration steps.
  ```

- **CI/CD and Automation**:
  ```
  Set up GitHub Actions or similar for [REPO_NAME] using shared_actions. Include Docker builds, tests, and deploys. Make it reusable for new projects.
  ```

- **Repo Merging Strategy**:
  ```
  Evaluate merging [REPO1_NAME] and [REPO2_NAME] (e.g., personal_games_clonehero and personal_games_coc). Pros/cons for DRY, impact on Docker setups, and steps to execute.
  ```

- **Project Template Creation**:
  ```
  Create a cookiecutter or similar template for new [TYPE, e.g., Python service] repos. Base it on shared_docker, shared_scripts, and MIT license. Ensure quick rollout.
  ```

## 5. Debugging and Troubleshooting Prompts
(Expanded with logging and multi-repo debug prompts.)

- **Debug Code Issue**:
  ```
  Debug this error in [REPO_NAME] ([LANGUAGE]): [ERROR_MESSAGE_AND_STACK_TRACE]. Consider Docker environment from shared_docker. Suggest fixes and preventive DRY measures.
  ```

- **Performance Optimization**:
  ```
  Optimize performance for [FEATURE_DESCRIPTION] in [REPO_NAME]. Profile and suggest improvements, leveraging shared repos for tools/utils.
  ```

- **Cross-Repo Dependency Check**:
  ```
  Check dependencies across [LIST_REPOS, e.g., fks_master, fks_transformer]. Flag vulnerabilities or outdated packages. Recommend updates via shared_scripts.
  ```

- **Logging Standardization**:
  ```
  Review logging in [REPO_NAME] and suggest standardization (e.g., structured logs). Integrate with shared_python or shared_scripts for DRY across services like fks_api and fks_worker.
  ```

- **Multi-Repo Debug Session**:
  ```
  Troubleshoot an issue spanning [LIST_REPOS, e.g., fks_engine and fks_execution]. Describe debug steps using Docker Compose from shared_docker.
  ```

## 6. Testing & Quality Assurance Prompts
(New category for unit/integration tests, focusing on DRY.)

- **Unit Test Generation**:
  ```
  Generate unit tests for this code in [REPO_NAME] ([LANGUAGE]): [PASTE_CODE_HERE]. Use frameworks like pytest (Python) or cargo test (Rust). Suggest shared test utils in a shared repo.
  ```

- **Integration Test Setup**:
  ```
  Set up integration tests for [REPO_NAME], involving [RELATED_REPOS, e.g., fks_data and fks_api]. Use Docker Compose for env, and shared_scripts for test runners.
  ```

- **Code Quality Check**:
  ```
  Suggest linters/formatters for [REPO_NAME] ([LANGUAGE]), e.g., black for Python. Integrate into CI via shared_actions for DRY enforcement.
  ```

- **Coverage Report Analysis**:
  ```
  Analyze test coverage for [REPO_NAME] and suggest improvements. Aim for DRY by sharing test patterns in shared repos.
  ```

## 7. Security & Compliance Prompts
(New category tailored to auth, data, and personal repos.)

- **Security Audit**:
  ```
  Perform a high-level security review of [REPO_NAME], focusing on [ASPECT, e.g., auth in fks_auth]. Suggest fixes using shared_python for crypto utils.
  ```

- **Compliance Check (e.g., MIT License)**:
  ```
  Ensure [REPO_NAME] complies with MIT license and open-source best practices. Check for proprietary code leaks in shared repos.
  ```

- **Vulnerability Scan**:
  ```
  Recommend tools to scan [REPO_NAME] for vulnerabilities, integrated with Docker and shared_actions.
  ```

- **Data Privacy in Personal Repos**:
  ```
  Review data handling in [REPO_NAME, e.g., personal_services_nextcloud]. Suggest DRY privacy utils from shared_schema.
  ```

## 8. Scaling & Deployment Prompts
(New category for growth to K8s and handling 32+ repos.)

- **Scaling Strategy**:
  ```
  Outline scaling for [REPO_NAME] from Docker Compose to K8s. Use shared_docker for base and shared_nginx for load balancing.
  ```

- **Deployment Pipeline**:
  ```
  Design a deployment pipeline for [LIST_REPOS, e.g., fks_web and fks_api]. Include rollback strategies and DRY via shared_scripts.
  ```

- **Monitoring Setup**:
  ```
  Suggest monitoring tools for [REPO_NAME], integrable with Docker/K8s. Share configs in a shared repo.
  ```

- **High-Availability Config**:
  ```
  Configure high-availability for services like fks_nginx and fks_worker. Prep for K8s clusters.
  ```

## 9. Collaboration & Version Control Prompts
(New category for team/multi-repo workflows.)

- **Branching Strategy**:
  ```
  Recommend a Git branching model for [REPO_NAME], considering integrations with shared repos. E.g., GitFlow or trunk-based.
  ```

- **Pull Request Template**:
  ```
  Create a PR template for [REPO_NAME], emphasizing DRY checks and links to shared docs.
  ```

- **Repo Sync Script**:
  ```
  Write a script to sync changes from shared_[REPO] to dependent repos like fks_[SERVICE]. Use shared_scripts.
  ```

- **Contributor Guidelines**:
  ```
  Generate CONTRIBUTING.md for [REPO_NAME], covering stack, DRY principles, and MIT license.
  ```

## 10. Miscellaneous Prompts
(Expanded with analytics and migration prompts.)

- **Documentation Generation**:
  ```
  Generate README.md or docs for [REPO_NAME], covering setup with Docker Compose, languages used, and links to shared repos. Keep it concise and DRY-focused.
  ```

- **Idea Brainstorming for New Features**:
  ```
  Brainstorm ideas for [FEATURE_TYPE, e.g., auth enhancements] across repos like fks_auth and fks_api. Prioritize DRY by using shared_rust or shared_python.
  ```

- **Repo Analytics**:
  ```
  Suggest ways to analyze activity across [LIST_REPOS, e.g., update times, commit frequency]. Use shell scripts from shared_scripts.
  ```

- **Legacy Migration (e.g., from personal_servers_)**:
  ```
  Plan migrating [REPO_NAME, e.g., personal_servers_freddy] to modern stack with Docker/K8s. Extract DRY parts to shared repos.
  ```