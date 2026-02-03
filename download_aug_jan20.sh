#!/bin/bash
# Download data from Aug 1, 2025 to Aug 31, 2025 and Jan 20, 2026

mkdir -p data

# August 2025
start_date="2025-08-01"
end_date="2025-08-31"
current="$start_date"

while [[ "$current" < "$end_date" ]] || [[ "$current" == "$end_date" ]]; do
    filename="subwaydatanyc_${current}_csv.tar.xz"
    url="https://subwaydata.nyc/data/${filename}"
    if [[ ! -f "data/${filename}" ]]; then
        echo "Downloading $filename..."
        curl -s -L -o "data/${filename}" "$url"
    else
        echo "Already have $filename"
    fi
    current=$(date -j -v+1d -f "%Y-%m-%d" "$current" "+%Y-%m-%d")
done

# Jan 20, 2026
filename="subwaydatanyc_2026-01-20_csv.tar.xz"
url="https://subwaydata.nyc/data/${filename}"
if [[ ! -f "data/${filename}" ]]; then
    echo "Downloading $filename..."
    curl -s -L -o "data/${filename}" "$url"
else
    echo "Already have $filename"
fi

echo "Download complete!"
