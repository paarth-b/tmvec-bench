#!/usr/bin/env python3
"""
Download CATH PDB files from the CATH database API.

Extracts domain IDs from a FASTA file and downloads corresponding PDB files,
skipping those that already exist.
"""

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, urlretrieve
from urllib.parse import urljoin


def extract_domain_ids(fasta_path):
    """Extract CATH domain IDs from FASTA file headers.
    
    Args:
        fasta_path: Path to the FASTA file
        
    Yields:
        Domain ID strings (e.g., '107lA00')
    """
    domain_pattern = re.compile(r'>cath\|[^|]+\|([^/]+)/')
    
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                match = domain_pattern.match(line)
                if match:
                    yield match.group(1)


def download_pdb(domain_id, output_dir, base_url, timeout=30, print_lock=None, progress=None):
    """Download a PDB file for a given domain ID.
    
    Args:
        domain_id: CATH domain ID (e.g., '107lA00')
        output_dir: Directory to save the PDB file
        base_url: Base URL for the CATH API
        timeout: Request timeout in seconds
        print_lock: Thread lock for thread-safe printing (optional)
        progress: Dict with 'current', 'total', 'successful', 'failed' counters (optional)
        
    Returns:
        Tuple of (domain_id, success: bool)
    """
    url = urljoin(base_url, f"{domain_id}.pdb")
    output_path = Path(output_dir) / f"{domain_id}.pdb"
    
    try:
        # Use urlretrieve for simple download
        urlretrieve(url, output_path)
        
        # Verify the file was downloaded and is not empty
        if output_path.exists() and output_path.stat().st_size > 0:
            success = True
        else:
            if output_path.exists():
                output_path.unlink()  # Remove empty file
            success = False
            
    except HTTPError as e:
        if print_lock:
            with print_lock:
                print(f"  HTTP error {e.code} for {domain_id}: {e.reason}", file=sys.stderr)
        success = False
    except URLError as e:
        if print_lock:
            with print_lock:
                print(f"  URL error for {domain_id}: {e.reason}", file=sys.stderr)
        success = False
    except Exception as e:
        if print_lock:
            with print_lock:
                print(f"  Unexpected error for {domain_id}: {e}", file=sys.stderr)
        if output_path.exists():
            output_path.unlink()  # Remove partial file
        success = False
    
    # Update progress counters if provided
    if progress is not None:
        with print_lock:
            progress['current'] += 1
            if success:
                progress['successful'] += 1
            else:
                progress['failed'] += 1
            current = progress['current']
            total = progress['total']
            print(f"[{current}/{total}] {domain_id}.pdb... {'✓' if success else '✗'}", flush=True)
    
    return (domain_id, success)


def main():
    parser = argparse.ArgumentParser(
        description="Download CATH PDB files from FASTA file"
    )
    parser.add_argument(
        '--fasta-file',
        type=str,
        default='/scratch/akeluska/ismb_submission/tmvec2/data/cath-domain-seqs-S100.fa',
        help='Path to CATH domain sequences FASTA file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/scratch/akeluska/ismb_submission/tmvec2/data/pdb/cath-s100',
        help='Output directory for PDB files (default: %(default)s)'
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default='https://www.cathdb.info/version/v4_4_0/api/rest/id/',
        help='Base URL for CATH API (default: %(default)s)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be downloaded without actually downloading'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=20,
        help='Number of concurrent download threads (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    fasta_path = Path(args.fasta_file)
    if not fasta_path.exists():
        print(f"Error: FASTA file not found: {fasta_path}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of existing PDB files
    existing_files = {f.stem for f in output_dir.glob('*.pdb')}
    print(f"Found {len(existing_files)} existing PDB files in {output_dir}")
    
    # Extract domain IDs and filter out existing ones
    domain_ids = list(extract_domain_ids(fasta_path))
    total_domains = len(domain_ids)
    
    to_download = [did for did in domain_ids if did not in existing_files]
    already_exist = total_domains - len(to_download)
    
    print(f"Total domains in FASTA: {total_domains}")
    print(f"Already downloaded: {already_exist}")
    print(f"To download: {len(to_download)}")
    
    if args.dry_run:
        print("\nDry run mode - would download:")
        for domain_id in to_download[:20]:  # Show first 20
            print(f"  {domain_id}.pdb")
        if len(to_download) > 20:
            print(f"  ... and {len(to_download) - 20} more")
        return
    
    if not to_download:
        print("All PDB files already downloaded!")
        return
    
    # Download missing files with multi-threading
    print(f"\nDownloading {len(to_download)} PDB files using {args.threads} threads...")
    
    # Thread-safe progress tracking
    print_lock = Lock()
    progress = {
        'current': 0,
        'total': len(to_download),
        'successful': 0,
        'failed': 0
    }
    
    # Use ThreadPoolExecutor for concurrent downloads
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Submit all download tasks
        futures = {
            executor.submit(
                download_pdb,
                domain_id,
                output_dir,
                args.base_url,
                print_lock=print_lock,
                progress=progress
            ): domain_id
            for domain_id in to_download
        }
        
        # Wait for all downloads to complete
        # Results are already printed by download_pdb with progress tracking
        for future in as_completed(futures):
            try:
                domain_id, success = future.result()
            except Exception as e:
                domain_id = futures[future]
                with print_lock:
                    print(f"  Exception for {domain_id}: {e}", file=sys.stderr)
                    progress['failed'] += 1
    
    print(f"\nDownload complete!")
    print(f"  Successful: {progress['successful']}")
    print(f"  Failed: {progress['failed']}")
    print(f"  Already existed: {already_exist}")
    print(f"  Total: {total_domains}")


if __name__ == '__main__':
    main()
