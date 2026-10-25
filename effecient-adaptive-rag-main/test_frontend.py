#!/usr/bin/env python3
"""Simple test to verify frontend imports work correctly."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from frontend.config import PAGE_CONFIG, CHAT_VIEW_HEIGHT, apply_theme
    from frontend.state.session import init_session
    from frontend.components.header import render_header
    from frontend.components.sidebar import render_sidebar
    from frontend.components.execution import render_execution_panel
    from frontend.components.chat import render_chat
    
    print("✓ All imports successful")
    print(f"✓ Page config: {PAGE_CONFIG['page_title']}")
    print(f"✓ Chat view height: {CHAT_VIEW_HEIGHT}")
    print("✓ Frontend modernization complete!")
    
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
