"""
Load all CSVs from data/raw/, deduplicate, verify counts.
Produces a pandas DataFrame `data` in the calling scope.
Expects DATA_RAW in the shared namespace (set by reproduce.py).
"""
import pandas as pd

NAME_MAP = {
    'anthropic/claude-opus-4-6@default': 'Opus 4.6',
    'anthropic/claude-opus-4-5@20251101': 'Opus 4.5',
    'anthropic/claude-opus-4-7@default': 'Opus 4.7',
    'anthropic/claude-opus-4-1@20250805': 'Opus 4.1',
    'anthropic/claude-sonnet-4-6@default': 'Sonnet 4.6',
    'anthropic/claude-sonnet-4-5@20250929': 'Sonnet 4.5',
    'anthropic/claude-sonnet-4@20250514': 'Sonnet 4',
    'anthropic/claude-haiku-4-5@20251001': 'Haiku 4.5',
    'deepseek-ai/deepseek-r1-0528': 'DeepSeek-R1',
    'deepseek-ai/deepseek-v3.2': 'DeepSeek V3.2',
    'deepseek-ai/deepseek-v3.1': 'DeepSeek V3.1',
    'google/gemini-3.1-pro-preview': 'Gemini 3.1 Pro',
    'google/gemini-3-flash-preview': 'Gemini 3 Flash',
    'google/gemini-2.5-flash': 'Gemini 2.5 Flash',
    'google/gemini-2.5-pro': 'Gemini 2.5 Pro',
    'google/gemini-3.1-flash-lite-preview': 'Gemini 3.1 FLite',
    'google/gemini-2.0-flash-lite': 'Gemini 2.0 FLite',
    'google/gemini-2.0-flash': 'Gemini 2.0 Flash',
    'google/gemma-4-31b': 'Gemma 4 31B',
    'google/gemma-3-27b': 'Gemma 3 27B',
    'google/gemma-3-12b': 'Gemma 3 12B',
    'google/gemma-3-4b': 'Gemma 3 4B',
    'google/gemma-3-1b': 'Gemma 3 1B',
    'openai/gpt-oss-20b': 'GPT-oss-20B',
    'openai/gpt-oss-120b': 'GPT-oss-120B',
    'openai/gpt-5.4-2026-03-05': 'GPT-5.4',
    'openai/gpt-5.4-mini-2026-03-17': 'GPT-5.4 mini',
    'openai/gpt-5.4-nano-2026-03-17': 'GPT-5.4 nano',
    'qwen/qwen3-next-80b-a3b-thinking': 'Qwen Think',
    'qwen/qwen3-next-80b-a3b-instruct': 'Qwen 80B Inst',
    'qwen/qwen3-coder-480b-a35b-instruct': 'Qwen Coder',
    'qwen/qwen3-235b-a22b-instruct-2507': 'Qwen 235B',
    'zai/glm-5': 'GLM-5',
}
FAMILY_MAP = {
    'Opus 4.7': 'Anthropic', 'Opus 4.6': 'Anthropic', 'Opus 4.5': 'Anthropic', 'Opus 4.1': 'Anthropic',
    'Sonnet 4.6': 'Anthropic', 'Sonnet 4.5': 'Anthropic', 'Sonnet 4': 'Anthropic', 'Haiku 4.5': 'Anthropic',
    'DeepSeek-R1': 'DeepSeek', 'DeepSeek V3.2': 'DeepSeek', 'DeepSeek V3.1': 'DeepSeek',
    'Gemini 3.1 Pro': 'G-Gemini', 'Gemini 3 Flash': 'G-Gemini', 'Gemini 2.5 Flash': 'G-Gemini',
    'Gemini 2.5 Pro': 'G-Gemini', 'Gemini 3.1 FLite': 'G-Gemini', 'Gemini 2.0 FLite': 'G-Gemini',
    'Gemini 2.0 Flash': 'G-Gemini',
    'Gemma 4 31B': 'G-Gemma', 'Gemma 3 27B': 'G-Gemma', 'Gemma 3 12B': 'G-Gemma',
    'Gemma 3 4B': 'G-Gemma', 'Gemma 3 1B': 'G-Gemma',
    'GPT-oss-20B': 'OpenAI', 'GPT-oss-120B': 'OpenAI', 'GPT-5.4': 'OpenAI',
    'GPT-5.4 mini': 'OpenAI', 'GPT-5.4 nano': 'OpenAI',
    'Qwen Think': 'Qwen', 'Qwen 80B Inst': 'Qwen', 'Qwen Coder': 'Qwen', 'Qwen 235B': 'Qwen',
    'GLM-5': 'Zhipu',
}

dfs = [pd.read_csv(f) for f in sorted(DATA_RAW.glob("*.csv"))]
raw = pd.concat(dfs, ignore_index=True)
data = raw.drop_duplicates(subset=['model', 'item_id', 'domain'], keep='first').copy()
data['is_correct'] = data['is_correct'].astype(str).str.lower().map({'true': 1, 'false': 0}).fillna(data['is_correct'])
data['is_correct'] = pd.to_numeric(data['is_correct'], errors='coerce').astype(int)
data['confidence'] = pd.to_numeric(data['confidence'], errors='coerce')
data = data.dropna(subset=['confidence', 'is_correct'])
data['model_short'] = data['model'].map(NAME_MAP)
data['family'] = data['model_short'].map(FAMILY_MAP)
assert data['model_short'].isna().sum() == 0, "Unmapped model(s) found"
assert data['model_short'].nunique() == 33, f"Expected 33 models, got {data['model_short'].nunique()}"
print(f"  Loaded {len(data):,} observations across {data['model_short'].nunique()} models")
