"""
Blockchain API Integration Module
Fetches real Bitcoin transaction data from Blockchain.info API
"""

import requests
import time
import pandas as pd
from datetime import datetime
import json

class BlockchainAPI:
    """Interface to fetch real blockchain data"""
    
    def __init__(self):
        self.base_url = "https://blockchain.info"
        self.session = requests.Session()
        
    def get_latest_block(self):
        """Get the latest block information"""
        try:
            url = f"{self.base_url}/latestblock"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error fetching latest block: {e}")
            return None
    
    def get_block_transactions(self, block_hash):
        """Get all transactions from a specific block"""
        try:
            url = f"{self.base_url}/rawblock/{block_hash}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error fetching block transactions: {e}")
            return None
    
    def get_transaction(self, tx_hash):
        """Get detailed information about a specific transaction"""
        try:
            url = f"{self.base_url}/rawtx/{tx_hash}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error fetching transaction: {e}")
            return None
    
    def get_address_info(self, address):
        """Get information about a Bitcoin address"""
        try:
            url = f"{self.base_url}/rawaddr/{address}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error fetching address info: {e}")
            return None
    
    def parse_transaction(self, tx_data):
        """Parse raw transaction data into our format"""
        try:
            # Calculate total input and output amounts
            total_input = sum([inp.get('prev_out', {}).get('value', 0) for inp in tx_data.get('inputs', [])])
            total_output = sum([out.get('value', 0) for out in tx_data.get('out', [])])
            
            # Convert from satoshis to BTC
            amount = total_output / 100000000
            fee = (total_input - total_output) / 100000000
            
            num_inputs = len(tx_data.get('inputs', []))
            num_outputs = len(tx_data.get('out', []))
            
            return {
                'transaction_id': tx_data.get('hash', 'unknown'),
                'amount': round(amount, 8),
                'num_inputs': num_inputs,
                'num_outputs': num_outputs,
                'fee': round(fee, 8) if fee > 0 else 0.0001,
                'timestamp': tx_data.get('time', int(time.time())),
                'block_height': tx_data.get('block_height', 0),
                'size': tx_data.get('size', 0)
            }
        except Exception as e:
            print(f"❌ Error parsing transaction: {e}")
            return None
    
    def fetch_latest_transactions(self, count=10):
        """Fetch latest transactions from the blockchain"""
        print(f"\n🔄 Fetching {count} latest transactions from Bitcoin blockchain...")
        
        try:
            # Get latest block
            latest_block = self.get_latest_block()
            if not latest_block:
                return []
            
            block_hash = latest_block.get('hash')
            block_height = latest_block.get('height')
            
            print(f"✅ Latest block: {block_height}")
            print(f"   Hash: {block_hash[:16]}...")
            
            # Get transactions from the block
            block_data = self.get_block_transactions(block_hash)
            if not block_data:
                return []
            
            transactions = block_data.get('tx', [])[:count]
            
            print(f"✅ Found {len(transactions)} transactions in block")
            
            # Parse transactions
            parsed_transactions = []
            for i, tx in enumerate(transactions, 1):
                print(f"   Processing transaction {i}/{len(transactions)}...", end='\r')
                parsed = self.parse_transaction(tx)
                if parsed:
                    parsed_transactions.append(parsed)
                time.sleep(0.1)  # Rate limiting
            
            print(f"\n✅ Successfully parsed {len(parsed_transactions)} transactions")
            return parsed_transactions
            
        except Exception as e:
            print(f"❌ Error fetching transactions: {e}")
            return []
    
    def save_transactions_to_csv(self, transactions, filename='live_transactions.csv'):
        """Save fetched transactions to CSV"""
        if not transactions:
            print("⚠️  No transactions to save")
            return
        
        df = pd.DataFrame(transactions)
        df.to_csv(filename, index=False)
        print(f"💾 Saved {len(transactions)} transactions to {filename}")
        return df


# Test function
def test_blockchain_api():
    """Test the blockchain API"""
    print("=" * 80)
    print("🧪 Testing Blockchain API Integration")
    print("=" * 80)
    
    api = BlockchainAPI()
    
    # Test 1: Get latest block
    print("\n📦 Test 1: Fetching latest block...")
    block = api.get_latest_block()
    if block:
        print(f"✅ Block Height: {block.get('height')}")
        print(f"✅ Block Hash: {block.get('hash')[:16]}...")
    
    # Test 2: Fetch transactions
    print("\n📦 Test 2: Fetching latest transactions...")
    transactions = api.fetch_latest_transactions(count=5)
    
    if transactions:
        print("\n📊 Sample transaction:")
        print(json.dumps(transactions[0], indent=2))
        
        # Save to CSV
        api.save_transactions_to_csv(transactions, 'test_live_transactions.csv')
    
    print("\n" + "=" * 80)
    print("✅ Blockchain API test complete!")
    print("=" * 80)


if __name__ == '__main__':
    test_blockchain_api()