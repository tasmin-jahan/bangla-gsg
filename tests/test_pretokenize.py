import sys
import os
import shutil
import tempfile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import pretokenize_and_pack

def test_pretokenize():
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Mock source configs with paths pointing to temp directories
        mock_data_dir = tmp_path / "data"
        mock_output_dir = tmp_path / "output"
        
        mock_configs = {}
        for source in ["bangla", "english"]:
            in_dir = mock_data_dir / source
            in_dir.mkdir(parents=True)
            out_dir = mock_output_dir / source
            mock_configs[source] = {
                "input_dir": in_dir,
                "output": out_dir
            }
            
            # Create a mock parquet file in the in_dir
            schema = pa.schema([
                ('text', pa.string()),
                ('source', pa.string()),
                ('source_type', pa.string()),
                ('language_region', pa.string()),
                ('word_count', pa.int64())
            ])
            
            # We want to create enough text to test tokenizer & saving.
            # "test_english_text " * 300 will generate around 600-900 tokens.
            # We'll put multiple documents to ensure it tests packing across documents.
            texts = [
                "আমার সোনার বাংলা আমি তোমায় ভালোবাসি। চিরদিন তোমার আকাশ তোমার বাতাস আমার প্রাণে বাজায় বাঁশি।",
                "hello world " * 500,  # about 1000 tokens
                "another document with some text " * 300, # about 1200 tokens
            ]
            
            data = [
                texts,
                ['source1'] * len(texts),
                ['type1'] * len(texts),
                ['bn'] * len(texts),
                [10] * len(texts)
            ]
            table = pa.Table.from_arrays(data, schema=schema)
            pq.write_table(table, in_dir / "shard_0000.parquet")
        
        # Override pretokenize_and_pack.SOURCE_CONFIGS and BATCH_TOKENS
        original_configs = pretokenize_and_pack.SOURCE_CONFIGS
        original_batch_tokens = pretokenize_and_pack.BATCH_TOKENS
        
        # Make BATCH_TOKENS small (e.g. 2048) so we trigger the flush during the loop
        pretokenize_and_pack.SOURCE_CONFIGS = mock_configs
        pretokenize_and_pack.BATCH_TOKENS = 2048
        
        # Load the real tokenizer (using saved/tokenizer directory)
        tokenizer = pretokenize_and_pack.load_tokenizer()
        
        try:
            # Run pretokenize_source on bangla
            tokens, docs = pretokenize_and_pack.pretokenize_source(
                "bangla",
                mock_configs["bangla"],
                tokenizer,
                max_tokens=None
            )
            print(f"Processed docs: {docs}, tokens: {tokens}")
            assert docs > 0
            assert tokens > 0
            
            # Verify outputs
            out_files = list(mock_configs["bangla"]["output"].glob("*.npy"))
            print(f"Generated files: {out_files}")
            assert len(out_files) > 0
            
            # Load the npy files to verify they have correct shape
            for f in out_files:
                arr = np.load(f)
                print(f"File shape: {arr.shape}")
                assert arr.ndim == 2
                assert arr.shape[1] == 2048
                
        finally:
            pretokenize_and_pack.SOURCE_CONFIGS = original_configs
            pretokenize_and_pack.BATCH_TOKENS = original_batch_tokens

if __name__ == "__main__":
    test_pretokenize()
    print("ALL PRETOKENIZATION TESTS PASSED! ✅")
