"""
Test script to demonstrate the improved language detection
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.process_dump import isEnglishDescription

# Test cases
test_cases = [
    # (description, expected_result, reason)
    ("", True, "Empty description (accepted)"),
    ("Short", True, "Too short to detect reliably"),
    ("This is a wonderful story about a brave hero who saves the day.", True, "Clear English text"),
    ("Un livre magnifique sur l'histoire de France", False, "French text"),
    ("这是一本关于中国历史的书", False, "Chinese text"),
    ("Это книга о русской истории", False, "Russian text (Cyrillic)"),
    ("A tale of adventure", True, "Short English description"),
    ("The author explores themes of love and loss in this compelling novel.", True, "Typical book description"),
    ("Lorem ipsum dolor sit amet", False, "Latin (no common English words)"),
    ("El autor explora temas de amor", False, "Spanish text"),
    ("Mystery thriller adventure", True, "Keywords only"),
]

print("Testing Language Detection")
print("=" * 80)
print()

correct = 0
total = len(test_cases)

for desc, expected, reason in test_cases:
    result = isEnglishDescription(desc)
    status = "PASS" if result == expected else "FAIL"
    if result == expected:
        correct += 1

    print(f"{status} {reason}")
    print(f"   Input: '{desc[:50]}{'...' if len(desc) > 50 else ''}'")
    print(f"   Expected: {expected}, Got: {result}")
    print()

print("=" * 80)
print(f"Results: {correct}/{total} correct ({correct/total*100:.1f}%)")
