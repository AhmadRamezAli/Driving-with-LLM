import re

def parse_actions(text: str) -> dict:
    """
    Extract accelerator, brake, and signed steering commands
    from a Vector-LM driving-policy explanation block.

    • Left turns  → negative steer value
    • Right turns → positive steer value
    • "Steer straight" → 0

    Returns
    -------
    dict
        {
          "accelerator_percent": float,  # 0.0 to 1.0
          "brake_percent":       float,  # 0.0 to 1.0  
          "steer_percent":       float,  # -1.0 to 1.0 (signed)
        }
    """
    accel = re.search(r"Accelerator\s+pedal\s+(\d+)%", text, re.I)
    brake = re.search(r"Brake\s+pedal\s+(\d+)%", text, re.I)

    steer_match = re.search(r"steer\s+(\d+)%\s+to\s+the\s+(right|left)", text, re.I)
    if steer_match:
        pct  = int(steer_match.group(1))
        sign = 1 if steer_match.group(2).lower() == "right" else -1
        steer_val = sign * pct
    elif re.search(r"steer\s+straight", text, re.I):
        steer_val = 0
    else:
        steer_val = None

    return {
        "accelerator_percent": float(accel.group(1)) / 100.0 if accel else 0.0,
        "brake_percent":       float(brake.group(1)) / 100.0 if brake else 0.0,
        "steer_percent":       float(steer_val) / 100.0 if steer_val is not None else 0.0,
    }
