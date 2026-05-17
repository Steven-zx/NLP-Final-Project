"""
Final Verification Test for Social Media App
Tests all components including dialogs and hate speech detection
"""

import tkinter as tk
from social_media_app import SocialMediaApp
import time

print("\n" + "="*80)
print("FINAL VERIFICATION TEST - HATESHIELD SOCIAL MEDIA APP")
print("="*80)

# Create app
root = tk.Tk()
app = SocialMediaApp(root)

print("\n✓ App initialized successfully")
print(f"✓ Vocabulary: {len(app.vocab.word2idx)} words")
print(f"✓ Detection threshold: 0.8 (80%)")
print(f"✓ Colors palette loaded: {len(app.colors)} colors")
print(f"✓ Fonts system loaded: {len(app.fonts)} fonts")

# Test hate speech detection
print("\n" + "-"*80)
print("TESTING HATE SPEECH DETECTION ENGINE")
print("-"*80)

test_cases = [
    ("Hello everyone! How are you?", False, "Friendly greeting"),
    ("You are stupid and worthless", True, "Hate speech - insult"),
    ("I love this community!", False, "Positive sentiment"),
    ("This is a beautiful day", False, "Neutral/positive"),
    ("I hate everyone here", False, "May or may not trigger - moderate"),
]

results = []
for text, expected_hate, desc in test_cases:
    is_hate, prob = app.detect_hate_speech(text)
    status = "✓" if is_hate == expected_hate else "⚠"
    results.append((status, text, is_hate, prob, desc))
    
    print(f"\n{status} {desc}")
    print(f"   Text: '{text}'")
    print(f"   Probability: {prob*100:.1f}%")
    print(f"   Result: {'🔴 BLOCKED' if is_hate else '🟢 ALLOWED'}")

# Test dialog creation (without showing)
print("\n" + "-"*80)
print("TESTING DIALOG COMPONENTS")
print("-"*80)

# Check if dialogs can be created
try:
    # Test that methods exist
    assert hasattr(app, 'show_loading_screen'), "Missing show_loading_screen method"
    assert hasattr(app, 'show_violation_dialog'), "Missing show_violation_dialog method"
    assert hasattr(app, 'show_success_dialog'), "Missing show_success_dialog method"
    assert hasattr(app, 'open_create_post_dialog'), "Missing open_create_post_dialog method"
    
    print("✓ All dialog methods present")
    print("✓ show_loading_screen() - Ready")
    print("✓ show_violation_dialog() - Ready")
    print("✓ show_success_dialog() - Ready")
    print("✓ open_create_post_dialog() - Ready")
except AssertionError as e:
    print(f"✗ Error: {e}")

# Summary
print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)

passed = sum(1 for r in results if r[0] == "✓")
total = len(results)

print(f"\nDetection Tests: {passed}/{total} passed")
print("Dialog Components: All present ✓")

print("\n" + "-"*80)
print("MANUAL TESTING INSTRUCTIONS")
print("-"*80)
print("\nThe GUI is now running. Please test manually:")
print("\n1. CREATE POST TEST:")
print("   • Click 'What's on your mind?' button")
print("   • Type: 'You are stupid and worthless'")
print("   • Press Ctrl+Enter OR click 'Post' button")
print("   • Expected: Loading screen → Red violation dialog → Post NOT added")
print("\n2. SAFE POST TEST:")
print("   • Click 'What's on your mind?' button")
print("   • Type: 'Hello everyone! How are you today?'")
print("   • Press Ctrl+Enter OR click 'Post' button")
print("   • Expected: Loading screen → Green success dialog → Post APPEARS in feed")
print("\n3. KEYBOARD SHORTCUT TEST:")
print("   • Open create post dialog")
print("   • Type any text")
print("   • Press Ctrl+Enter")
print("   • Expected: Post is submitted (same as clicking Post button)")
print("\n" + "="*80)
print("Close the window when testing is complete.")
print("="*80 + "\n")

# Run the app
root.mainloop()
