
def temperature_control(temp, occupied, old_mode):
    if not occupied:
        new_mode = "Off"
    elif temp < 18:
        new_mode = "Heat"
    elif temp > 26:
        new_mode = "Cool"
    else:
        new_mode = "Eco"

    if new_mode != old_mode:
        send_command = True
    else:
        send_command = False

    return new_mode, send_command


test_cases = [
    {
        "name": "Case 1 - Heating Required",
        "temperature": 16,
        "occupied": True,
        "previous_mode": None
    },
    {
        "name": "Case 2 - Same Heating Mode",
        "temperature": 17,
        "occupied": True,
        "previous_mode": "Heat"
    },
    {
        "name": "Case 3 - Minimum Temperature Limit",
        "temperature": 18,
        "occupied": True,
        "previous_mode": "Heat"
    },
    {
        "name": "Case 4 - Maximum Temperature Limit",
        "temperature": 26,
        "occupied": True,
        "previous_mode": "Eco"
    },
    {
        "name": "Case 5 - Cooling Required",
        "temperature": 28,
        "occupied": True,
        "previous_mode": "Eco"
    },
    {
        "name": "Case 6 - Room Not Occupied",
        "temperature": 22,
        "occupied": False,
        "previous_mode": "Eco"
    },
    {
        "name": "Case 7 - Already Switched Off",
        "temperature": 22,
        "occupied": False,
        "previous_mode": "Off"
    }
]


for case in test_cases:
    mode, send_command = temperature_control(
        case["temperature"],
        case["occupied"],
        case["previous_mode"]
    )

    print(case["name"])
    print("Temperature:", case["temperature"])
    print("Room Occupied:", case["occupied"])
    print("Previous Mode:", case["previous_mode"])
    print("New Mode:", mode)

    if send_command:
        print("Action: Command Sent")
    else:
        print("Action: No Command Needed")

    print("-" * 45)
