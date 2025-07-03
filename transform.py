import pandas as pd
from datetime import timedelta
import json
from tqdm import tqdm

INPUT_CSV = "chat_output.csv"
OUTPUT_JSON = "rag_conversations.json"
HISTORY_LENGTH = 3                # How much message history to include before your response
CHUNK_TIMEOUT_MINUTES = 10       # Gap to split conversations
MIN_RESPONSE_WORDS = 2           # Ignore very short responses (like "อืม", "555")

# Load and sort
df = pd.read_csv(INPUT_CSV, parse_dates=["DateTime"])
df = df.sort_values("DateTime").reset_index(drop=True)

# Group into time-based chunks
chunks = []
current_chunk = []
prev_time = df.loc[0, "DateTime"]

for _, row in tqdm(df.iterrows(), "Split data from CSV into chunks."):
    time_diff = row["DateTime"] - prev_time
    if time_diff > timedelta(minutes=CHUNK_TIMEOUT_MINUTES):
        if current_chunk:
            chunks.append(current_chunk)
        current_chunk = []
    current_chunk.append(row)
    prev_time = row["DateTime"]
if current_chunk:
    chunks.append(current_chunk)

# Helper to normalize message history
def normalize(history):
    return " ".join(h.strip().lower() for h in history[-HISTORY_LENGTH:])

# Generate pairs
rag_data = []
seen_keys = set()
used_responses = set()

print(f"🔄 Processing {len(chunks)} chat sessions...")

for chunk in tqdm(chunks, desc="🔧 Generating RAG pairs", unit="session"):
    messages = [f'{r["Sender"]}: {r["Message"]}' for r in chunk]
    for i in range(len(messages) - 1):
        if messages[i].startswith("cream♡:"):
            for j in range(i + 1, min(i + 5, len(messages))):
                if messages[j].startswith("frank:"):
                    response = messages[j].strip()
                    if len(response.split()) < MIN_RESPONSE_WORDS:
                        continue  # skip very short responses
                    if response in used_responses:
                        continue  # already used
                    history_start = max(0, i - HISTORY_LENGTH + 1)
                    history = [m.strip() for m in messages[history_start:j]]
                    key = (normalize(history), response.lower())
                    if key not in seen_keys:
                        rag_data.append({
                            "history": history,
                            "response": response
                        })
                        seen_keys.add(key)
                        used_responses.add(response)
                    break

# Save output
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(rag_data, f, ensure_ascii=False, indent=2)

print(f"\n✅ Done! Exported {len(rag_data)} unique RAG entries to {OUTPUT_JSON}")
