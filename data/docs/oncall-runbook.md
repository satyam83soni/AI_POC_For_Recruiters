# Skillpath Backend On-Call Runbook

## Severity levels

SEV1: the learning platform or payments are down for all users. Page the on-call engineer immediately; target acknowledgement is 5 minutes.
SEV2: a major feature (AI tutor, video playback, certificates) is degraded. Acknowledge within 15 minutes.
SEV3: a minor bug with a workaround. Handle in business hours.

## AI tutor latency alert

The alert fires when p95 tutor latency exceeds 8 seconds for 10 minutes. First checks, in order: the LLM provider status page, the rate-limit error count on the tutor dashboard, database connection pool usage, and the vector search p95 latency.

If the LLM provider is rate limiting us, enable the degraded mode flag `tutor.short_answers`, which lowers max output tokens and cuts load.

## Database connection pool exhausted

Symptoms: requests hang and then fail with pool timeout errors, while database CPU stays low. Usual cause: a code path that opens a session and never closes it, or a session held open across a slow external call. Restarting the service clears it temporarily; the real fix is finding the leak.

## Escalation

If a SEV1 is not mitigated within 30 minutes, escalate to the engineering manager on call. Payments incidents are always escalated to the payments lead, whatever the severity.
