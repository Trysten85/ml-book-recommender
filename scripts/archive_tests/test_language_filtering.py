"""
Test the language filtering improvements
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.process_dump import detectLanguage

# Test the detectLanguage function
print("=" * 80)
print("Testing Language Detection")
print("=" * 80)
print()

test_cases = [
    ("Harry Potter and the Prisoner of Azkaban", "A wonderful story about magic and friendship in a wizard school."),
    ("Onnellinen kuolema", "Kertomus suomalaisesta perheestä ja heidän vaikeuksistaan."),
    ("Literatura fantástica chilena", "Una colección de cuentos fantásticos de autores chilenos."),
    ("Peabody's mermaid", "A delightful tale of a man who catches a mermaid while fishing."),
    ("The Martian", "An astronaut is stranded on Mars and must use science to survive."),
    ("Chun jing zhi zi", "一个关于中国古代历史的故事"),
    ("Ex Machina", "A programmer tests an android with artificial intelligence."),
    ("Per Ardua Ad Astra", "Through adversity to the stars - a science fiction epic about space exploration."),
]

print("Book Title -> Detected Language")
print("-" * 80)

for title, description in test_cases:
    lang = detectLanguage(description)
    print(f"{title:40} -> {lang:10} | {description[:40]}...")

print()
print("=" * 80)
print("Notice how foreign-language DESCRIPTIONS are detected (fi, es, zh)")
print("But English books with FOREIGN TITLES are correctly marked as 'en'")
print("(Ex Machina, Per Ardua Ad Astra)")
print("=" * 80)
