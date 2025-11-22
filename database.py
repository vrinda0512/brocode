import sqlite3
import json
from datetime import datetime

class WatchlistDB:
    def __init__(self, db_file='watchlist.db'):
        self.db_file = db_file
        self.init_db()
    
    def init_db(self):
        """Create watchlist table if it doesn't exist"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                value TEXT NOT NULL UNIQUE,
                risk_level TEXT NOT NULL,
                reason TEXT,
                tags TEXT,
                activity_count INTEGER DEFAULT 0,
                added_date TEXT NOT NULL,
                last_seen TEXT,
                total_volume REAL DEFAULT 0,
                status TEXT DEFAULT 'active',
                notes TEXT
            )
        ''')
        
        # ✅ ADD: Create transactions table on init
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT UNIQUE,
                amount REAL,
                num_inputs INTEGER,
                num_outputs INTEGER,
                fee REAL,
                risk_score INTEGER,
                prediction TEXT,
                alert_level TEXT,
                timestamp TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully!")
    
    def add_to_watchlist(self, type, value, risk_level, reason, tags=None, notes=None):
        """Add new item to watchlist"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO watchlist (type, value, risk_level, reason, tags, added_date, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (type, value, risk_level, reason, json.dumps(tags or []), datetime.now().isoformat(), notes))
            
            conn.commit()
            conn.close()
            return {"success": True, "message": "Added to watchlist successfully!"}
        except sqlite3.IntegrityError:
            conn.close()
            return {"success": False, "message": "Item already exists in watchlist!"}
    
    def get_all_watchlist(self):
        """Get all watchlist items"""
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM watchlist ORDER BY added_date DESC')
        rows = cursor.fetchall()
        
        # Convert to list of dictionaries
        watchlist = []
        for row in rows:
            item = dict(row)
            item['tags'] = json.loads(item['tags']) if item['tags'] else []
            watchlist.append(item)
        
        conn.close()
        return watchlist
    
    def check_watchlist(self, value):
        """Check if value exists in watchlist"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM watchlist WHERE value = ? AND status = "active"', (value,))
        result = cursor.fetchone()
        
        conn.close()
        return result is not None
    
    def update_activity(self, value, amount=0):
        """Update activity count and last seen for watchlist item"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE watchlist 
            SET activity_count = activity_count + 1,
                last_seen = ?,
                total_volume = total_volume + ?
            WHERE value = ?
        ''', (datetime.now().isoformat(), amount, value))
        
        conn.commit()
        conn.close()
    
    def remove_from_watchlist(self, watchlist_id):
        """Remove item from watchlist"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM watchlist WHERE id = ?', (watchlist_id,))
        
        conn.commit()
        conn.close()
        return {"success": True, "message": "Removed from watchlist!"}
    
    def get_watchlist_stats(self):
        """Get watchlist statistics"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Total count
        cursor.execute('SELECT COUNT(*) FROM watchlist WHERE status = "active"')
        total = cursor.fetchone()[0]
        
        # Count by risk level
        cursor.execute('SELECT risk_level, COUNT(*) FROM watchlist WHERE status = "active" GROUP BY risk_level')
        by_risk = dict(cursor.fetchall())
        
        # Recent activity (last 7 days)
        cursor.execute('''
            SELECT COUNT(*) FROM watchlist 
            WHERE status = "active" 
            AND last_seen >= date('now', '-7 days')
        ''')
        active_week = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total": total,
            "critical": by_risk.get("CRITICAL", 0),
            "high": by_risk.get("HIGH", 0),
            "medium": by_risk.get("MEDIUM", 0),
            "low": by_risk.get("LOW", 0),
            "active_this_week": active_week
        }

    def add_transaction_result(self, result):
        """Add analyzed transaction result to database"""
        try:
            # ✅ FIXED: Changed self.db_path to self.db_file
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Insert transaction (table already created in init_db)
            cursor.execute('''
                INSERT OR REPLACE INTO transactions 
                (transaction_id, amount, num_inputs, num_outputs, fee, risk_score, prediction, alert_level, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['transaction_id'],
                result['amount'],
                result['num_inputs'],
                result['num_outputs'],
                result['fee'],
                result['risk_score'],
                result['prediction'],
                result['alert_level'],
                result['timestamp']
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error adding transaction result: {e}")
            return False
    
    def get_recent_transactions(self, limit=100):
        """Get recent analyzed transactions"""
        try:
            # ✅ FIXED: Changed self.db_path to self.db_file
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT transaction_id, amount, risk_score, prediction, alert_level, timestamp
                FROM transactions
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [{
                'transaction_id': row[0],
                'amount': row[1],
                'risk_score': row[2],
                'prediction': row[3],
                'alert_level': row[4],
                'timestamp': row[5]
            } for row in rows]
            
        except Exception as e:
            print(f"❌ Error getting transactions: {e}")
            return []

# Initialize database when module is imported
db = WatchlistDB()