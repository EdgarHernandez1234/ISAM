This bundle creates the first two layers of the new operator stack:

1. `MainMenuApp`
2. `ManualOperatorApp` state machine

It keeps the old PS4 + Jetson structure conceptually intact, but uses placeholder adapters for now so you can build the file structure before wiring live controller input.

## Files

- `run_operator_menu.py` — entrypoint
- `rover_operator/enums.py` — control, mission, safety, and preset enums
- `rover_operator/models.py` — run state, mission flags, event records
- `rover_operator/config.py` — shared config and topic names
- `rover_operator/adapters.py` — placeholder controller / rover / arm / dashboard adapters
- `rover_operator/event_logger.py` — CSV event logger
- `rover_operator/manual_operator_app.py` — stage 2 operator state machine
- `rover_operator/main_menu.py` — stage 1 menu

## What works now

- menu layer
- manual operator session startup / shutdown
- BASE / ARM / HALT mode handling
- speed ladder state
- event logging to CSV
- mission-phase transitions and mission flags
- autonomous driver stub

## What is intentionally deferred

- live PS4 controller input
- live rover output transport
- real arm preset execution
- main dashboard
- autonomy implementation

## Suggested placement in the shared workspace

Host:

```bash
~/sim_vendor/space_robotics_gz_envs/scripts/operator_app/
```

Container (same shared volume):

```bash
/root/ws/scripts/operator_app/
```

## Run after copying into the workspace

```bash
cd /root/ws/scripts/operator_app
python3 run_operator_menu.py
```

The current implementation uses console placeholders, so `Manual Operator` will run until `Ctrl+C` returns you to the menu.
