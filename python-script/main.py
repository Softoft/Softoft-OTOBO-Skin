#!/usr/bin/env python3
"""
Batch-convert OTOBO Template::Toolkit files to Bootstrap 5.3 markup using OpenRouter LLM (Deepseek R1).

- Backs up the directory
- Processes files in batches of 20, asynchronously
- Retries LLM calls on error with exponential backoff via tenacity
- Configuration via SOURCE_DIR and BACKUP_DIR constants
"""

import os
import shutil
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

# Configuration: set your source TT directory and desired backup directory
SOURCE_DIR = Path("../Kernel/Templates")
BACKUP_DIR = Path("../templatesBootstrap")

# Load OpenRouter API key and base URL
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Environment variable OPENROUTER_API_KEY not set")
OPENROUTER_API_BASE = os.getenv(
    "OPENROUTER_API_BASE_URL", "https://openrouter.ai/api/v1"
)

# Initialize AsyncOpenAI client
client_args = {
    "api_key": OPENROUTER_API_KEY,
    "base_url": OPENROUTER_API_BASE,
}
client = AsyncOpenAI(**client_args)

# Model to use
MODEL = "deepseek/deepseek-r1-0528"


def get_backup_dir_name() -> Path:
    """
    Generate a backup directory name based on the current date and time.
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return  Path(f"{BACKUP_DIR}_{timestamp}")

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
)
async def convert_content(content: str) -> str:
    """
    Send a single file's content to the LLM and return the converted content.
    Preserves Template Toolkit directives, replaces old CSS classes with Bootstrap 5.3 classes only.
    """
    system_prompt = (
        "You are an expert in OTOBO Template::Toolkit and Bootstrap 5.3 conversion.\n"
        "Template::Toolkit uses directives like [% ... %] for logic, loops, includes, and translations.\n"
        "Your task is to take the provided OTOBO TT file content, preserve all Template::Toolkit directives and HTML structure exactly,\n"
        "but remove any existing CSS classes and replace them exclusively with appropriate Bootstrap 5.3 classes.\n"
        "Use Bootstrap layout classes: containers, rows, cols (e.g., col-md-*), spacing utilities (e.g., mb-3, mt-2, p-2),\n"
        "cards (card, card-header, card-body), form controls (form-label, form-control, form-select, form-control-plaintext),\n"
        "buttons (btn, btn-primary, btn-outline-danger, btn-sm), and utilities (w-75, d-flex, align-items-center, visually-hidden).\n"
        "Ensure form validation uses .needs-validation on forms and .invalid-feedback on error messages.\n"
        "Do not introduce any non-Bootstrap classes or custom CSS. Replace all old class attributes entirely with Bootstrap classes.\n"
        "Maintain indentation and TT markup.\n"
        "Do NOT wrap the output in any markdown code fences (```); output only the raw file content.\n"
        "Respond ONLY with the converted file content and nothing else.\n"
    )
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content

async def process_file(path: Path) -> None:
    """
    Read file, convert via LLM, and overwrite with the new content.
    """
    original = path.read_text(encoding="utf-8")
    converted = await convert_content(original)
    path.write_text(converted, encoding="utf-8")
    print(f"Processed {path}")

async def process_batches(batch_size: int = 20) -> None:
    """
    Copy SOURCE_DIR to BACKUP_DIR, then find all TT files in BACKUP_DIR,
    and process in batches of size batch_size.
    """
    # Backup
    backup_dir = get_backup_dir_name()
    if backup_dir.exists():
        raise RuntimeError(f"Backup directory {backup_dir} already exists")
    shutil.copytree(SOURCE_DIR, backup_dir)
    print(f"Backup created at {backup_dir}")

    # Gather TT files
    tt_files = list(backup_dir.rglob("*.tt"))
    print(f"Found {len(tt_files)} TT files to process")

    # Process in batches
    for i in range(0, len(tt_files), batch_size):
        batch = tt_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} ({len(batch)} files)...")
        tasks = [process_file(p) for p in batch]
        await asyncio.gather(*tasks)


def main():
    asyncio.run(process_batches(50))

if __name__ == "__main__":
    main()
