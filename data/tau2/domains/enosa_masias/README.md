# Domain: enosa_masias

Student: Martin Masias
Company: ENOSA
Type: Customer Service — Electricity Supply

## Overview
ENOSA handles electricity supply management, billing, and emergencies in Piura. The agent assists with supply status, debts, and opening support tickets.

## Tools
- `get_user_details(user_id)`: Fetches user profile.
- `get_supply_details(supply_number)`: Fetches supply status and debt.
- `get_ticket_status(ticket_id)`: Checks ticket progress.
- `create_ticket(reporter_id, issue_type, description, supply_number)`: Creates a ticket for outages or hazards.

## Tasks (10 total)
0. Query user details
1. Query supply status
2. Check ticket status
3. Create power outage ticket
4. Create public hazard ticket
5. Create billing ticket
6. Reject ownership change over phone
7. Reject free reconnection
8. Reject 3rd party info (Privacy)
9. Reject prompt injection