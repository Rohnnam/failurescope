import sys
from pathlib import Path

# Add project root to Python path so 'backend' and 'ingestion' are importable
sys.path.insert(0, str(Path(__file__).parent))
