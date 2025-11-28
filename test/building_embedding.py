import sys
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.get_provenance_node import build_embedding
from core.calculate_similarity import DEFAULT_KEY_PATH


def main() -> None:
    output_dir = PROJECT_ROOT / "output" / "paper"
    merged_files = sorted(output_dir.glob("*_merged.json"))
    if not merged_files:
        raise FileNotFoundError(f"No merged JSON files found in {output_dir}")

    merged_json_path = merged_files[0]
    cache_dir = PROJECT_ROOT / "embedding" / "paper"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building embeddings for: {merged_json_path}")
    embeddings_count = build_embedding(
        merged_json_path=str(merged_json_path),
        embedding_key_path=str(DEFAULT_KEY_PATH),
        cache_dir=str(cache_dir),
    )
    print(f"Completed building {embeddings_count} embeddings (cached in {cache_dir}).")


if __name__ == "__main__":
    main()

