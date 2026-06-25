From your screenshot, you are inside:

~/Downloads/ADU/eventhub

That looks like your repo/work folder. Do not run this test inside the repo. First move to a clean temporary folder from Git Bash.

Run these commands one by one.

1. Go outside repo and create temp folder

cd ~
mkdir -p eh-connectivity-test
cd eh-connectivity-test
pwd

You should now see something like:

/c/Users/your-user/eh-connectivity-test

2. Create virtual environment

python --version || py -3 --version

Then run:

python -m venv .venv || py -3 -m venv .venv

Activate it:

source .venv/Scripts/activate

If activation works, your terminal should show something like:

(.venv)

3. Install Event Hub package

python -m pip install --upgrade pip
python -m pip install azure-eventhub

4. Add connection string temporarily

This will not save the connection string in any file. It only stays in this Git Bash session.

read -s -r -p "Paste Event Hub connection string: " EVENT_HUB_CONNECTION_STR; echo
export EVENT_HUB_CONNECTION_STR

Now add Event Hub name:

read -r -p "Paste Event Hub name: " EVENT_HUB_NAME
export EVENT_HUB_NAME

Important: EVENT_HUB_NAME should be the actual Event Hub name, not the namespace.

5. Confirm values are loaded without printing secret

Run this:

python - <<'PY'
import os
cs = os.environ.get("EVENT_HUB_CONNECTION_STR", "")
eh = os.environ.get("EVENT_HUB_NAME", "")
print("Connection string loaded:", bool(cs))
print("Contains EntityPath:", "EntityPath=" in cs)
print("Event Hub name loaded:", bool(eh))
PY

Expected output:

Connection string loaded: True
Contains EntityPath: True/False
Event Hub name loaded: True

6. Create test script

Run this full command:

cat > send_eventhub_test.py <<'PY'
import os
import json
import uuid
from datetime import datetime, timezone
from azure.eventhub import EventHubProducerClient, EventData, TransportType
conn_str = os.environ.get("EVENT_HUB_CONNECTION_STR")
eventhub_name = os.environ.get("EVENT_HUB_NAME")
if not conn_str:
    raise SystemExit("EVENT_HUB_CONNECTION_STR is not set.")
client_args = {
    "conn_str": conn_str,
    "transport_type": TransportType.AmqpOverWebsocket,
}
if "EntityPath=" not in conn_str:
    if not eventhub_name:
        raise SystemExit("EVENT_HUB_NAME is required because connection string does not contain EntityPath.")
    client_args["eventhub_name"] = eventhub_name
test_id = f"manual-eh-test-{uuid.uuid4()}"
payload = {
    "testId": test_id,
    "source": "manual-cross-tenant-eventhub-test",
    "message": "Testing Event Hub connectivity from standalone temp script",
    "timestampUtc": datetime.now(timezone.utc).isoformat()
}
print("Connecting to Event Hub...")
producer = EventHubProducerClient.from_connection_string(**client_args)
with producer:
    batch = producer.create_batch()
    batch.add(EventData(json.dumps(payload)))
    producer.send_batch(batch)
print("Event sent successfully.")
print("Test ID:", test_id)
PY

7. Run the test

python send_eventhub_test.py

If successful, you should see:

Connecting to Event Hub...
Event sent successfully.
Test ID: manual-eh-test-...

That means the connection string is valid and your machine can send data to the Event Hub.

8. Verify in Azure Portal

Ask the person who has portal access, or check yourself:

Event Hub Namespace
→ Event Hubs
→ your event hub
→ Metrics
→ Incoming Messages

Run the script once or twice and check whether Incoming Messages increased.

9. Clean up after testing

After test is done:

unset EVENT_HUB_CONNECTION_STR
unset EVENT_HUB_NAME
deactivate
cd ~
rm -rf eh-connectivity-test

.


python - <<'PY'
import os

cs = os.environ.get("EVENT_HUB_CONNECTION_STR", "")
eh = os.environ.get("EVENT_HUB_NAME", "")

print("Connection string loaded:", bool(cs))
print("Contains EntityPath:", "EntityPath=" in cs)
print("Event Hub name loaded:", bool(eh))
print("Event Hub name:", eh)
PY

