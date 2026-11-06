def marks(marks, documents, attendance, assignment):
    if marks < 70:
        return "FAIL", "Marks are less than 70"

    elif not documents:
        return "FAIL", "Documents are missing"

    elif attendance < 75:
        return "FAIL", "Attendance requirement is not fulfilled"

    elif not assignment:
        return "FAIL", "Assignment has not been submitted"

    else:
        return "PASS", "Student meets all requirements"


test_cases = [
    {
        "name": "Case 1 - Student Passed",
        "input": (85, True, 90, True),
        "expected": "PASS"
    },
    {
        "name": "Case 2 - Low Attendance",
        "input": (82, True, 70, True),
        "expected": "FAIL"
    },
    {
        "name": "Case 3 - Assignment Missing",
        "input": (88, True, 90, False),
        "expected": "FAIL"
    },
    {
        "name": "Case 4 - Minimum Marks",
        "input": (70, True, 80, True),
        "expected": "PASS"
    }
]

for case in test_cases:
    marks, documents, attendance, assignment = case["input"]

    result, reason = marks_check(marks, documents, attendance, assignment)

    print(case["name"])
    print("Marks:", marks)
    print("Documents:", documents)
    print("Attendance:", attendance, "%")
    print("Assignment Submitted:", assignment)
    print("Expected:", case["expected"])
    print("Actual:", result)
    print("Reason:", reason)
    print("-" * 40)