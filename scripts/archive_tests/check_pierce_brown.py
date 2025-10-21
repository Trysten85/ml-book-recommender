import pandas as pd

books = pd.read_pickle('data/processed/books.pkl')

# Find all Pierce Brown books (solo author)
pierce_solo = books[books['authorNames'] == 'Pierce Brown']

print(f'Books with ONLY Pierce Brown as author: {len(pierce_solo)}\n')

for idx, row in pierce_solo.iterrows():
    print(f'Title: {row["title"]}')
    subjects = str(row.get('subjects', 'N/A'))
    print(f'Subjects: {subjects[:150]}...' if len(subjects) > 150 else f'Subjects: {subjects}')
    desc_len = len(str(row.get('description', ''))) if pd.notna(row.get('description')) else 0
    print(f'Description: {desc_len} chars')
    print()
