"""Download the official UCI dataset; choose the output path with --output."""
import argparse
import io
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = 'https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/raw/data.csv'))
    args = parser.parse_args()
    with urlopen(URL, timeout=60) as response:
        archive = response.read()
    with ZipFile(io.BytesIO(archive)) as zf:
        data = zf.read('data.csv')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(f'Saved {len(data)} bytes to {args.output}')


if __name__ == '__main__':
    main()
