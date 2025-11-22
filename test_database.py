from database import db

# Test 1: Add items to watchlist
print("🧪 Testing Database...")

# Add suspicious address
result1 = db.add_to_watchlist(
    type="address",
    value="1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
    risk_level="CRITICAL",
    reason="Known scammer address",
    tags=["scammer", "high-volume"],
    notes="Associated with exchange hack"
)
print("Test 1:", result1)

# Add another address
result2 = db.add_to_watchlist(
    type="address",
    value="3J98t1WpEZ73CNmYviecrnyiWrnqRhWNLy",
    risk_level="HIGH",
    reason="Mixing service detected",
    tags=["mixer", "suspicious"],
    notes="Multiple fraud patterns"
)
print("Test 2:", result2)

# Get all watchlist items
print("\n📋 All Watchlist Items:")
all_items = db.get_all_watchlist()
for item in all_items:
    print(f"  - {item['value']} | Risk: {item['risk_level']} | Reason: {item['reason']}")

# Check if address is on watchlist
print("\n🔍 Checking Watchlist:")
is_watched = db.check_watchlist("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")
print(f"  Address on watchlist: {is_watched}")

# Update activity
print("\n📊 Updating Activity...")
db.update_activity("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", amount=10.5)
print("  Activity updated!")

# Get stats
print("\n📈 Watchlist Statistics:")
stats = db.get_watchlist_stats()
print(f"  Total: {stats['total']}")
print(f"  Critical: {stats['critical']}")
print(f"  High: {stats['high']}")
print(f"  Active this week: {stats['active_this_week']}")

print("\n✅ Database tests completed!")