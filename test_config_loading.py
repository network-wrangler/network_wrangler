#!/usr/bin/env python3
"""Test script to verify YAML configuration loading works correctly."""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import directly from configs module to avoid triggering full package imports
from network_wrangler.configs.__init__ import load_wrangler_config

def test_yaml_config():
    """Test loading the example YAML configuration file."""
    config_path = Path("examples/stpaul/wrangler.config.yml")
    
    print(f"Loading configuration from: {config_path}")
    config = load_wrangler_config(config_path)
    
    print("✓ YAML config loaded successfully!")
    print(f"\nConfiguration values:")
    print(f"  ML_OFFSET_METERS: {config.MODEL_ROADWAY.ML_OFFSET_METERS}")
    print(f"  ML_LINK_ID_METHOD: {config.IDS.ML_LINK_ID_METHOD}")
    print(f"  ML_LINK_ID_SCALAR: {config.IDS.ML_LINK_ID_SCALAR}")
    print(f"  ML_NODE_ID_METHOD: {config.IDS.ML_NODE_ID_METHOD}")
    print(f"  EXISTING_VALUE_CONFLICT: {config.EDITS.EXISTING_VALUE_CONFLICT}")
    print(f"  OVERWRITE_SCOPED: {config.EDITS.OVERWRITE_SCOPED}")
    print(f"  CPU csv read speed: {config.CPU.EST_PD_READ_SPEED['csv']}")
    
    # Verify values match what's in the YAML file
    assert config.MODEL_ROADWAY.ML_OFFSET_METERS == -10, "ML_OFFSET_METERS should be -10"
    assert config.IDS.ML_LINK_ID_METHOD == "scalar", "ML_LINK_ID_METHOD should be 'scalar'"
    assert config.IDS.ML_LINK_ID_SCALAR == 3000000, "ML_LINK_ID_SCALAR should be 3000000"
    assert config.IDS.ML_NODE_ID_METHOD == "range", "ML_NODE_ID_METHOD should be 'range'"
    assert config.EDITS.EXISTING_VALUE_CONFLICT == "warn", "EXISTING_VALUE_CONFLICT should be 'warn'"
    
    print("\n✓ All configuration values verified correctly!")
    return True

if __name__ == "__main__":
    try:
        test_yaml_config()
        print("\n✅ YAML configuration loading test PASSED")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

