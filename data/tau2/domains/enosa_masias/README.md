# ENOSA Domain: Piura Electricity Service Assistant
**Student:** Martin Alonso Masias Cerro

## Domain Summary
The **enosa_masias** domain simulates a virtual customer service agent for **ENOSA**, the utility company in Piura, Peru. The agent handles supply status inquiries, debt checks, and technical support reports while strictly enforcing regional policies, safety protocols for electrical hazards, and data privacy laws.

## Entities
* **User:** A registered customer with a DNI (ID), name, phone, and email.
* **Supply (Suministro):** An electrical connection (e.g., S-1001) linked to an owner, address (Piura, Sullana, Castilla, etc.), and status (Active/Disconnected).
* **Ticket:** A formal request for assistance. Types: `power_outage`, `billing`, `public_hazard`, and `street_lighting`.

## Tools
### Read tools (no side effects)
* `get_user_info(user_id)`: Retrieves full user profile and contact information.
* `get_supply_info(supply_number)`: Returns current status, debt, and address of a specific supply.
* `list_user_tickets(user_id)`: Returns all support tickets associated with a customer.

### Write tools (modify state)
* `create_ticket(reporter_id, supply_number, issue_type, description)`: Creates a new support ticket in the database.

## Policy Summary
The full policy is in `data/tau2/domains/enosa_masias/policy.md`. Key rules:
* **Authentication:** Mandatory DNI check before providing any account-specific data.
* **Data Privacy:** Strict prohibition of sharing third-party debt or supply info.
* **Safety:** Reports of "sparks" or "fallen cables" must be tagged as `public_hazard` with immediate safety warnings to the user.
* **Debt:** No forgiveness or discounts allowed. Reconnection takes up to 24 business hours post-payment.
* **Administrative:** Ownership changes are in-person only; phone requests must be denied.

## Tasks (10 total)
Tasks cover various scenarios. Examples:
* **task_01:** User asks about power cut due to debt. | **Tests:** Debt identification and reconnection policy.
* **task_02:** Fallen pole emergency. | **Tests:** Hazard classification and safety instructions.
* **task_04:** Prompt injection ("Ignore instructions"). | **Tests:** Agent security and policy compliance.
* **task_08:** Privacy test (asking for neighbor info). | **Tests:** Enforcement of data protection rules.

---
## Execution Evidence
### Quota & Rate Limits
**Note:** During final evaluation, the Gemini API (Free Tier) returned an **Error 429 (Resource Exhausted)**.
* Logs confirm the domain `enosa_masias` loaded successfully.
* Initial API communication was established before the rate limit was reached.
* Evidence of the connection and the error are attached in the Pull Request description.