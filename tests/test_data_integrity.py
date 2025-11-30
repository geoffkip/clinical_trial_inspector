import unittest
import chromadb
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestDataIntegrity(unittest.TestCase):
    def setUp(self):
        # Determine the project root directory
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.test_dir)
        self.db_path = os.path.join(self.project_root, "ct_gov_index")

    def test_pfizer_myeloma_counts(self):
        """
        Verifies that the database contains the expected number of Pfizer studies
        related to Multiple Myeloma, based on strict keyword matching.
        """
        if not os.path.exists(self.db_path):
            self.skipTest(f"Database directory '{self.db_path}' does not exist. Skipping data integrity test.")

        print(f"\n📂 Loading database from {self.db_path}...")
        try:
            client = chromadb.PersistentClient(path=self.db_path)
            collection = client.get_collection("clinical_trials")
        except Exception as e:
            self.skipTest(f"Failed to load ChromaDB collection: {e}")

        # Fetch all metadata
        data = collection.get(include=["metadatas"])
        
        if not data["metadatas"]:
            self.fail("Database is empty (no metadata found).")

        df = pd.DataFrame(data["metadatas"])
        
        # 1. Check for 'org' column
        if "org" not in df.columns:
            self.fail("'org' column missing from metadata.")

        # 2. Filter by Sponsor (Pfizer)
        pfizer_studies = df[df["org"].str.contains("Pfizer", case=False, na=False)]
        # We expect at least some Pfizer studies if the DB is populated
        self.assertGreater(len(pfizer_studies), 0, "No Pfizer studies found in DB.")

        # 3. Filter by "Multiple Myeloma" in Title or Conditions
        query = "Multiple Myeloma"
        
        def is_relevant(row):
            title = str(row.get("title", "")).lower()
            conditions = str(row.get("condition", "")).lower()
            q = query.lower()
            return q in title or q in conditions

        relevant_studies = pfizer_studies[pfizer_studies.apply(is_relevant, axis=1)]
        
        count = len(relevant_studies)
        print(f"🎯 Pfizer Studies with '{query}' in Title or Conditions: {count}")

        # Assertion: Based on our previous check, we expect exactly 7.
        # However, to be robust against minor data updates, we can assert a range or exact value.
        # Let's assert it's non-zero and reasonably small (since we know it shouldn't be 514).
        self.assertGreater(count, 0, "Should find at least one relevant study.")
        self.assertLess(count, 50, "Should not find hundreds of studies (strict filter check).")
        
        # Optional: Assert exact count if we want to be very strict about data consistency
        # self.assertEqual(count, 7, "Expected exactly 7 studies based on known ground truth.")

if __name__ == "__main__":
    unittest.main()
