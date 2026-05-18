# MCP Policy

MCP tools should extend agent capability without bypassing engineering safety.

## Use MCP For

- Read-only repository, issue, PR, and CI inspection.
- Browser smoke tests and screenshots.
- Database schema inspection through read-only roles.
- Documentation search.
- Internal tools with explicit schemas and audit logs.

## Be Careful With

- Tools that can write files, deploy, merge, delete, or change permissions.
- Tools that expose secrets, customer data, or private messages.
- Tools whose descriptions include unreviewed instructions.
- Tools installed from unknown marketplaces.

## Default Permission Model

1. Prefer read-only tools first.
2. Require explicit user intent for writes.
3. Avoid broad tokens; use least privilege.
4. Log important write actions in the final response.
5. Never paste secrets into prompts or docs.

## Tool Selection

Use the narrowest tool that answers the task:

- GitHub connector for PR/issues/reviews.
- Browser for local UI smoke tests.
- Database MCP for schema/query inspection.
- Slack/Teams only when the user asks for workspace communication.

If a task can be done safely with local files and tests, do that before adding a
new MCP dependency.
