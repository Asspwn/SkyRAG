#!/usr/bin/env python3
"""Simple display test without Jupyter"""

import requests
import pandas as pd
import re
from io import StringIO

# Search
response = requests.post("http://localhost:8080/api/users/testuser/search",
                        json={"query": "What is SFT", "top_k": 1})
result = response.json()['results'][0]

# Display text
print("\n" + "="*80)
print("TEXT CONTENT")
print("="*80)
print(result['content'][:500] + "...")

# Extract and display tables
def extract_tables(text):
    table_pattern = r'<table>.*?</table>'
    tables = re.findall(table_pattern, text, re.DOTALL)

    for idx, table_html in enumerate(tables, 1):
        try:
            df = pd.read_html(StringIO(table_html))[0]
            print(f"\n{'='*80}")
            print(f"TABLE {idx}")
            print("="*80)
            print(df.to_string())
        except Exception as e:
            print(f"Could not parse table {idx}: {e}")

extract_tables(result['content'])

# Display images
print(f"\n{'='*80}")
print("IMAGES")
print("="*80)
for idx, img_path in enumerate(result['images'], 1):
    print(f"{idx}. {img_path}")

print(f"\n✅ Found {len(result['images'])} image(s)")
print(f"📊 Score: {result['score']:.4f}")
